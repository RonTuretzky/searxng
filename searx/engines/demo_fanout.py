# SPDX-License-Identifier: AGPL-3.0-or-later
"""Demo engine for the ``online_fanout`` engine type.

This engine fans one user query out into ``fanout_count`` parallel HTTP
requests against ``base_url``. Each sub-request appends a different ``page``
parameter so the demo doubles as a parallel-pagination example. Responses are
expected to be JSON arrays of result dicts; each dict must have ``title`` and
``url`` keys.

settings.yml::

    - name: demo fanout
      engine: demo_fanout
      shortcut: df
      base_url: https://example.com/api/search?q={query}&page={page}
      fanout_count: 8
      network:
        max_connections: 256
        max_keepalive_connections: 64
"""

from urllib.parse import quote_plus

from searx.network import Request

engine_type = "online_fanout"
categories = ["general"]
paging = False
time_range_support = False

# engine config (overridable in settings.yml)
base_url = "https://example.com/api/search?q={query}&page={page}"
fanout_count = 8

about = {
    "website": None,
    "wikidata_id": None,
    "official_api_documentation": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "JSON",
}


def init(engine_settings):  # pylint: disable=unused-argument
    if fanout_count < 1:
        raise ValueError("fanout_count must be >= 1")


def requests(query, params):  # pylint: disable=unused-argument
    """Build N parallel sub-requests for one user query.

    The default fan-out unit is *pagination* — same query, pages 1..N. Override
    this function (or ``base_url``) in your own engine module to fan out by
    sub-query, shard, or backend instead.
    """
    q = quote_plus(query)
    return [
        Request.get(base_url.format(query=q, page=page))
        for page in range(1, fanout_count + 1)
    ]


def response(resp):
    """Parse one sub-response. Called once per request returned by ``requests``."""
    results = []
    try:
        payload = resp.json()
    except ValueError:
        return results

    if not isinstance(payload, list):
        return results

    for item in payload:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        if not title or not url:
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "content": item.get("content", ""),
            }
        )
    return results
