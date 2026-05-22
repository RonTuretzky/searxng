# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,invalid-name,protected-access
"""Tests for :py:class:`searx.search.processors.online_fanout.OnlineFanoutProcessor`.

Two layers of coverage:

1. **Correctness** — :py:class:`TestOnlineFanoutProcessor` mocks
   :py:func:`searx.network.multi_requests` and asserts that the processor calls
   the engine's :py:func:`response` once per successful response, counts
   per-sub-request errors instead of failing the whole engine, and merges
   results in order.

2. **Performance** — :py:class:`TestOnlineFanoutPerf` spins up a local
   threaded HTTP server that sleeps before responding and compares serial
   :py:func:`searx.network.get` against parallel
   :py:func:`searx.network.multi_requests`. Parallel wall time must be
   substantially less than serial wall time — that is the whole point of the
   processor.
"""

import http.server
import threading
import time
import types
from timeit import default_timer

import httpx
from mock import patch

import searx.network
from searx.network import Request
from searx.network.network import Network
from searx.search.processors.online import default_request_params
from searx.search.processors.online_fanout import OnlineFanoutProcessor

from tests import SearxTestCase


def _make_response(status_code: int = 200, text: str = "{}", url: str = "https://example.com/") -> httpx.Response:
    return httpx.Response(status_code=status_code, text=text, request=httpx.Request("GET", url))


def _make_engine(name: str = "fake-fanout", request_list=None):
    """Build a stand-in module-like engine object for the processor to drive."""
    eng = types.SimpleNamespace()
    eng.name = name
    eng.timeout = 5.0
    eng.requests = lambda query, params: list(request_list or [])
    # response() is set by individual tests
    return eng


class TestOnlineFanoutProcessor(SearxTestCase):

    def _build_processor(self, engine):
        # OnlineFanoutProcessor inherits OnlineProcessor; we only exercise
        # _search_basic here so we don't need full processor lifecycle setup.
        proc = OnlineFanoutProcessor.__new__(OnlineFanoutProcessor)
        proc.engine = engine
        return proc

    def test_no_requests_returns_none(self):
        engine = _make_engine(request_list=[])
        proc = self._build_processor(engine)
        result = proc._search_basic("anything", {})  # type: ignore[arg-type]
        self.assertIsNone(result)

    def test_all_responses_parsed_and_merged(self):
        urls = [f"https://example.com/{i}" for i in range(4)]
        engine = _make_engine(request_list=[Request.get(u) for u in urls])
        # response() returns one result per sub-request, labeled by URL path.
        engine.response = lambda resp: [{"title": str(resp.url.path), "url": str(resp.url)}]
        proc = self._build_processor(engine)

        fake_responses = [_make_response(url=u) for u in urls]
        with patch.object(searx.network, "multi_requests", return_value=fake_responses) as mr:
            results = proc._search_basic("test", {})  # type: ignore[arg-type]

        mr.assert_called_once()
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 4)  # type: ignore[arg-type]
        # order matches request_list (which is the contract of multi_requests)
        self.assertEqual([r["url"] for r in results], urls)  # type: ignore[index]

    def test_subrequest_exception_is_counted_not_raised(self):
        urls = [f"https://example.com/{i}" for i in range(3)]
        engine = _make_engine(request_list=[Request.get(u) for u in urls])
        engine.response = lambda resp: [{"title": "ok", "url": str(resp.url)}]
        proc = self._build_processor(engine)

        # middle sub-request fails
        fake_responses = [
            _make_response(url=urls[0]),
            httpx.TimeoutException("slow shard", request=None),
            _make_response(url=urls[2]),
        ]
        with patch.object(searx.network, "multi_requests", return_value=fake_responses), patch(
            "searx.search.processors.online_fanout.count_error"
        ) as count_error_mock:
            results = proc._search_basic("test", {})  # type: ignore[arg-type]

        # whole engine still produces results from the two successes
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)  # type: ignore[arg-type]
        # one secondary error counted for the failed sub-request
        self.assertEqual(count_error_mock.call_count, 1)
        _args, kwargs = count_error_mock.call_args
        self.assertTrue(kwargs.get("secondary"))

    def test_standard_engine_loops_pageno(self):
        """An engine with the standard ``request(query, params)`` API (no
        ``requests()`` method) gets driven by the processor across N pagenos."""
        seen_pagenos: list[int] = []

        def standard_request(query, params):  # pylint: disable=unused-argument
            seen_pagenos.append(params["pageno"])
            params["url"] = f"https://example.com/?q={query}&start={(params['pageno'] - 1) * 10}"

        engine = types.SimpleNamespace()
        engine.name = "fake-standard"
        engine.timeout = 5.0
        engine.fanout_pages = 5
        engine.request = standard_request
        engine.response = lambda resp: [{"title": "ok", "url": str(resp.url)}]

        proc = OnlineFanoutProcessor.__new__(OnlineFanoutProcessor)
        proc.engine = engine

        # capture the request list multi_requests is invoked with
        captured: dict[str, list] = {}

        def fake_multi(request_list):
            captured["request_list"] = list(request_list)
            return [_make_response(url=r.url) for r in request_list]

        params = {**default_request_params(), "pageno": 1}
        with patch.object(searx.network, "multi_requests", side_effect=fake_multi):
            results = proc._search_basic("hello", params)  # type: ignore[arg-type]

        self.assertEqual(seen_pagenos, [1, 2, 3, 4, 5])
        self.assertEqual(len(captured["request_list"]), 5)
        # each sub-request hits a distinct ``start=`` offset
        start_offsets = sorted(
            int(r.url.split("start=")[-1]) for r in captured["request_list"]
        )
        self.assertEqual(start_offsets, [0, 10, 20, 30, 40])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)  # type: ignore[arg-type]

    def test_all_subrequests_fail_reraises_for_base_handler(self):
        """When every sub-request fails, the processor must re-raise so
        :py:meth:`OnlineProcessor.search` can run its SSL/timeout/captcha
        suspension policy. Partial failures (some successes) are *not*
        re-raised — those are downgraded to secondary error counters."""
        urls = [f"https://example.com/{i}" for i in range(2)]
        engine = _make_engine(request_list=[Request.get(u) for u in urls])
        engine.response = lambda resp: []
        proc = self._build_processor(engine)

        first_exc = httpx.TimeoutException("a", request=None)
        fake_responses = [first_exc, httpx.TimeoutException("b", request=None)]
        with patch.object(searx.network, "multi_requests", return_value=fake_responses), patch(
            "searx.search.processors.online_fanout.count_error"
        ):
            with self.assertRaises(httpx.TimeoutException) as ctx:
                proc._search_basic("test", {})  # type: ignore[arg-type]

        # the first exception (in request-order) is the one re-raised, so the
        # base handler sees a representative failure rather than a generic one.
        self.assertIs(ctx.exception, first_exc)

    def test_standard_engine_response_sees_per_page_params(self):
        """The pagination path must hand each response its own per-page
        ``search_params`` — not the outer ``params`` — so engines parsing
        results read the correct ``pageno``/context for that response."""
        received_pagenos: list[int] = []

        def standard_request(query, params):  # pylint: disable=unused-argument
            params["url"] = f"https://example.com/?page={params['pageno']}"

        def response_fn(resp):
            # the engine's response() reads pageno off resp.search_params
            received_pagenos.append(resp.search_params["pageno"])
            return [{"title": "ok", "url": str(resp.url)}]

        engine = types.SimpleNamespace()
        engine.name = "fake-paged"
        engine.timeout = 5.0
        engine.fanout_pages = 3
        engine.request = standard_request
        engine.response = response_fn

        proc = OnlineFanoutProcessor.__new__(OnlineFanoutProcessor)
        proc.engine = engine

        def fake_multi(request_list):
            return [_make_response(url=r.url) for r in request_list]

        params = {**default_request_params(), "pageno": 1}
        with patch.object(searx.network, "multi_requests", side_effect=fake_multi):
            proc._search_basic("hi", params)  # type: ignore[arg-type]

        # each response saw the pageno that produced its sub-request, not the
        # outer params' pageno=1
        self.assertEqual(received_pagenos, [1, 2, 3])


# ---------------------------------------------------------------------------
# Perf test
# ---------------------------------------------------------------------------


class _SleepyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that sleeps ``DELAY`` seconds before responding.

    The sleep simulates a slow remote — what fan-out is designed to mask.
    """

    DELAY = 0.2

    def do_GET(self):  # noqa: N802 (stdlib API)
        time.sleep(self.DELAY)
        body = b'[{"title": "hello", "url": "https://example.com/x", "content": ""}]'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # silence noisy default access log
        pass


class TestOnlineFanoutPerf(SearxTestCase):
    """Measure that ``multi_requests`` actually parallelizes — i.e. that the
    fan-out processor delivers the speedup it claims.

    This is the proof that fan-out (which the processor wraps) beats the
    serialized one-request-at-a-time model.
    """

    SUB_REQUESTS = 8
    """Number of parallel sub-requests issued in the perf comparison."""

    @classmethod
    def setUpClass(cls):
        # Bind to port 0 directly — the OS picks a free port atomically with
        # the bind, so there's no TOCTOU window between selection and use.
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SleepyHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        super().setUp()
        # Give this thread a sane network context + timeout budget so
        # searx.network.* functions work outside the normal processor flow.
        # The default network has HTTP disabled — we need a private Network with
        # enable_http=True and a generous pool so the parallel run isn't choked.
        searx.network.set_timeout_for_thread(10.0, start_time=default_timer())
        searx.network.reset_time_for_thread()
        self._http_network = Network(
            enable_http=True,
            max_connections=32,
            max_keepalive_connections=32,
        )
        searx.network.THREADLOCAL.network = self._http_network

    def tearDown(self):
        # release the private network's clients
        try:
            import asyncio

            from searx.network.client import get_loop

            fut = asyncio.run_coroutine_threadsafe(self._http_network.aclose(), get_loop())
            fut.result(5.0)
        finally:
            searx.network.THREADLOCAL.__dict__.pop("network", None)
            super().tearDown()

    def _url(self, i: int) -> str:
        return f"http://127.0.0.1:{self.port}/{i}"

    def test_multi_requests_is_faster_than_serial(self):
        # serial baseline — what a normal single-request engine looks like, repeated.
        start = default_timer()
        for i in range(self.SUB_REQUESTS):
            resp = searx.network.get(self._url(i))
            self.assertEqual(resp.status_code, 200)
        serial_elapsed = default_timer() - start

        # parallel fan-out — what the OnlineFanoutProcessor will use internally.
        request_list = [Request.get(self._url(i)) for i in range(self.SUB_REQUESTS)]
        start = default_timer()
        responses = searx.network.multi_requests(request_list)
        parallel_elapsed = default_timer() - start

        self.assertEqual(len(responses), self.SUB_REQUESTS)
        for resp in responses:
            self.assertFalse(isinstance(resp, Exception), "multi_requests returned an exception")
            self.assertEqual(resp.status_code, 200)  # type: ignore[union-attr]

        # Each request sleeps DELAY seconds server-side. Serial wall time ≈
        # SUB_REQUESTS * DELAY. Parallel wall time should be roughly DELAY
        # (plus overhead). Require at least a 3x speedup with N=8, DELAY=0.2s —
        # well clear of measurement noise on any reasonable machine.
        speedup = serial_elapsed / parallel_elapsed
        self.assertGreater(
            speedup,
            3.0,
            f"expected fan-out to be at least 3x faster than serial; "
            f"got serial={serial_elapsed:.3f}s parallel={parallel_elapsed:.3f}s speedup={speedup:.2f}x",
        )

    def test_fanout_processor_end_to_end_speedup(self):
        """Drive the actual processor against the sleepy server to confirm the
        full integration (engine.requests -> multi_requests -> engine.response)
        runs in parallel."""
        urls = [self._url(i) for i in range(self.SUB_REQUESTS)]
        engine = _make_engine(name="perf-fanout", request_list=[Request.get(u) for u in urls])
        engine.response = lambda resp: [{"title": "t", "url": str(resp.url), "content": ""}]

        proc = OnlineFanoutProcessor.__new__(OnlineFanoutProcessor)
        proc.engine = engine

        start = default_timer()
        results = proc._search_basic("any", {})  # type: ignore[arg-type]
        elapsed = default_timer() - start

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), self.SUB_REQUESTS)  # type: ignore[arg-type]

        # With DELAY=0.2s and SUB_REQUESTS=8, serial would take ~1.6s. Parallel
        # should finish well under half of that.
        serial_estimate = _SleepyHandler.DELAY * self.SUB_REQUESTS
        self.assertLess(
            elapsed,
            serial_estimate / 2,
            f"processor wall time {elapsed:.3f}s not significantly under serial estimate "
            f"{serial_estimate:.3f}s — fan-out may be serializing",
        )
