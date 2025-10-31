from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import json
import typing as t
from threading import Lock
import logging

    import argparse
    import sys
from typing import Any, List, Dict, Optional
import asyncio
logger = logging.getLogger("extractor_stats")

class ExtractorStats:
    """
    Tracks extractor performance per domain and persists to JSON.

    Methods:
        update(domain, extractor, success, elapsed): update stats
        get_ordered_extractors(domain, extractors): order by score
        reset(): clear all stats
        get_stats(domain=None, extractor=None): get stats
        export_json(): export as JSON string
        import_json(data): import from JSON string
        get_global_summary(): summary across all domains
    """
    def __init__(self, stats_file: str):
        
    """__init__ function."""
self.stats_file = stats_file
        self.stats = self._load()
        self._lock = Lock()

    def _load(self) -> dict:
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning(f"[ExtractorStats] Failed to load stats: {e}")
        return {}

    def save(self) -> None:
        with self._lock:
            try:
                with open(self.stats_file, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(self.stats, f, indent=2)
                logger.info(f"[ExtractorStats] Stats saved to {self.stats_file}")
            except Exception as e:
                logger.warning(f"[ExtractorStats] Could not save stats: {e}")

    def update(self, domain: str, extractor: str, success: bool, elapsed: float) -> None:
        with self._lock:
            d = self.stats.setdefault(domain, {}).setdefault(extractor, {'successes': 0, 'failures': 0, 'total_time': 0.0, 'runs': 0})
            if success:
                d['successes'] += 1
            else:
                d['failures'] += 1
            d['total_time'] += elapsed
            d['runs'] += 1
            logger.debug(f"[ExtractorStats] Updated: {domain} | {extractor} | success={success} | elapsed={elapsed:.3f}s")
            self.save()

    def get_ordered_extractors(self, domain: str, extractors: t.List) -> t.List:
        """Order extractors by historical score for this domain."""
        stats = self.stats.get(domain, {})
        def score(extractor) -> Any:
            d = stats.get(extractor.name, None)
            if not d or d['runs'] == 0:
                return 0  # unknown extractors go last
            avg_time = d['total_time'] / d['runs']
            return d['successes'] / (d['failures'] + 1) - avg_time * 0.1
        ordered = sorted(extractors, key=score, reverse=True)
        logger.debug(f"[ExtractorStats] Ordered extractors for {domain}: {[e.name for e in ordered]}")
        return ordered

    def reset(self) -> None:
        with self._lock:
            self.stats = {}
            self.save()
            logger.info("[ExtractorStats] Stats reset.")

    def get_stats(self, domain: t.Optional[str] = None, extractor: t.Optional[str] = None) -> dict:
        """Get stats for a domain, extractor, or all."""
        if domain and extractor:
            result = self.stats.get(domain, {}).get(extractor, {})
        elif domain:
            result = self.stats.get(domain, {})
        elif extractor:
            result = {d: v.get(extractor, {}) for d, v in self.stats.items() if extractor in v}
        else:
            result = self.stats
        logger.debug(f"[ExtractorStats] get_stats(domain={domain}, extractor={extractor}): {result}")
        return result

    def export_json(self) -> str:
        """Export stats as JSON string."""
        data = json.dumps(self.stats, indent=2)
        logger.debug(f"[ExtractorStats] export_json: {data[:200]}...")
        return data

    def import_json(self, data: str) -> None:
        """Import stats from JSON string. Validates structure."""
        try:
            obj = json.loads(data)
            if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
                with self._lock:
                    self.stats = obj
                    self.save()
                    logger.info("[ExtractorStats] Stats imported from JSON.")
            else:
                logger.warning("[ExtractorStats] Invalid structure in import_json.")
        except Exception as e:
            logger.warning(f"[ExtractorStats] Failed to import stats: {e}")

    def get_global_summary(self) -> dict:
        """Return a summary of total successes, failures, avg time per extractor across all domains."""
        summary = {}
        for domain, extractors in self.stats.items():
            for name, d in extractors.items():
                s = summary.setdefault(name, {'successes': 0, 'failures': 0, 'total_time': 0.0, 'runs': 0})
                s['successes'] += d.get('successes', 0)
                s['failures'] += d.get('failures', 0)
                s['total_time'] += d.get('total_time', 0.0)
                s['runs'] += d.get('runs', 0)
        for name, s in summary.items():
            s['avg_time'] = s['total_time'] / s['runs'] if s['runs'] else None
        logger.debug(f"[ExtractorStats] get_global_summary: {summary}")
        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExtractorStats CLI")
    parser.add_argument('--file', type=str, default='extractor_stats.json', help='Stats file path')
    parser.add_argument('--reset', action='store_true', help='Reset all stats')
    parser.add_argument('--export', action='store_true', help='Export stats as JSON')
    parser.add_argument('--import_json', type=str, help='Import stats from JSON string or file')
    parser.add_argument('--summary', action='store_true', help='Show global summary')
    parser.add_argument('--domain', type=str, help='Show stats for domain')
    parser.add_argument('--extractor', type=str, help='Show stats for extractor')
    args = parser.parse_args()
    stats = ExtractorStats(args.file)
    if args.reset:
        stats.reset()
        print("Stats reset.")
        sys.exit(0)
    if args.import_json:
        if os.path.isfile(args.import_json):
            with open(args.import_json, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                data = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        else:
            data = args.import_json
        stats.import_json(data)
        print("Stats imported.")
        sys.exit(0)
    if args.export:
        print(stats.export_json())
        sys.exit(0)
    if args.summary:
        print(json.dumps(stats.get_global_summary(), indent=2))
        sys.exit(0)
    if args.domain or args.extractor:
        print(json.dumps(stats.get_stats(domain=args.domain, extractor=args.extractor), indent=2))
        sys.exit(0)
    # Default: print all stats
    print(json.dumps(stats.get_stats(), indent=2)) 