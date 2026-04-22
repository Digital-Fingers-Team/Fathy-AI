import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import Dataset, IterableDataset, load_dataset

logger = logging.getLogger(__name__)


DATASETS: Dict[str, Dict[str, Any]] = {
    "wikipedia_ar": {
        "source": "wikimedia/wikipedia",
        "subset": "20231101.ar",
        "split": "train",
        "language": "ar",
        "license": "CC-BY-SA-3.0",
        "estimated_tokens": 3_200_000_000,
        "type": "encyclopedia",
    },
    "wikipedia_en": {
        "source": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "language": "en",
        "license": "CC-BY-SA-3.0",
        "estimated_tokens": 25_000_000_000,
        "type": "encyclopedia",
    },
    "oscar_ar": {
        "source": "oscar-corpus/OSCAR-2301",
        "subset": "ar",
        "split": "train",
        "language": "ar",
        "license": "ODC-BY",
        "estimated_tokens": 12_500_000_000,
        "type": "web_crawl",
    },
    "oscar_en": {
        "source": "oscar-corpus/OSCAR-2301",
        "subset": "en",
        "split": "train",
        "language": "en",
        "license": "ODC-BY",
        "estimated_tokens": 150_000_000_000,
        "type": "web_crawl",
    },
    "c4_en": {
        "source": "allenai/c4",
        "subset": "en",
        "split": "train",
        "language": "en",
        "license": "ODC-BY",
        "estimated_tokens": 156_000_000_000,
        "type": "web_crawl",
    },
    "mc4_ar": {
        "source": "mc4",
        "subset": "ar",
        "split": "train",
        "language": "ar",
        "license": "ODC-BY",
        "estimated_tokens": 8_200_000_000,
        "type": "web_crawl",
    },
}


class DataCollector:
    """Collects and normalizes datasets into a unified JSONL format."""

    def __init__(self, datasets_catalog: Optional[Dict[str, Dict[str, Any]]] = None):
        self.datasets_catalog = datasets_catalog or DATASETS

    def _normalize_record(self, dataset_name: str, sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        cfg = self.datasets_catalog[dataset_name]
        source = cfg["source"]
        language = cfg["language"]

        if dataset_name.startswith("wikipedia"):
            text = sample.get("text")
            sample_id = sample.get("id") or sample.get("url") or f"{dataset_name}-{idx}"
            metadata = {
                "title": sample.get("title"),
                "url": sample.get("url"),
            }
        elif dataset_name.startswith("oscar") or dataset_name.startswith("mc4") or dataset_name.startswith("c4"):
            text = sample.get("text")
            sample_id = sample.get("id") or sample.get("url") or f"{dataset_name}-{idx}"
            metadata = {
                "url": sample.get("url"),
                "timestamp": sample.get("timestamp"),
            }
        else:
            text = sample.get("text") or sample.get("content") or sample.get("document")
            sample_id = sample.get("id") or f"{dataset_name}-{idx}"
            metadata = {k: v for k, v in sample.items() if k not in {"text", "content", "document"}}

        if not text or not isinstance(text, str):
            return None

        return {
            "id": str(sample_id),
            "source": source,
            "language": language,
            "text": text,
            "metadata": metadata,
        }

    def _iter_dataset(self, ds: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(ds, (Dataset, IterableDataset)):
            for row in ds:
                yield row
        else:
            for row in ds:
                yield row

    def _update_license_manifest(self, output_dir: Path, dataset_name: str) -> None:
        manifest_path = output_dir / "licenses.json"
        cfg = self.datasets_catalog[dataset_name]

        manifest: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "datasets": {}}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("licenses.json is malformed, recreating file")

        manifest.setdefault("datasets", {})[dataset_name] = {
            "source": cfg["source"],
            "license": cfg["license"],
            "language": cfg["language"],
            "type": cfg["type"],
            "estimated_tokens": cfg["estimated_tokens"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def download(self, dataset_name: str, output_dir: str, max_samples: Optional[int] = None) -> Path:
        if dataset_name not in self.datasets_catalog:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {sorted(self.datasets_catalog)}")

        cfg = self.datasets_catalog[dataset_name]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Loading %s from %s", dataset_name, cfg["source"])
        ds = load_dataset(cfg["source"], cfg.get("subset"), split=cfg.get("split", "train"))

        file_path = output_path / f"{dataset_name}.jsonl"
        written = 0
        with file_path.open("w", encoding="utf-8") as f:
            for idx, sample in enumerate(self._iter_dataset(ds)):
                if max_samples is not None and written >= max_samples:
                    break
                record = self._normalize_record(dataset_name, sample, idx)
                if record is None:
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        self._update_license_manifest(output_path, dataset_name)
        logger.info("Downloaded %s samples for %s -> %s", written, dataset_name, file_path)
        return file_path

    def download_all(self, output_dir: str, datasets: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        selected = list(datasets) if datasets is not None else list(self.datasets_catalog.keys())
        results: Dict[str, Dict[str, Any]] = {}

        total = len(selected)
        for i, name in enumerate(selected, start=1):
            logger.info("[%s/%s] Downloading dataset: %s", i, total, name)
            try:
                file_path = self.download(name, output_dir)
                results[name] = {"status": "ok", "path": str(file_path)}
                logger.info("[%s/%s] Completed: %s", i, total, name)
            except Exception as exc:
                logger.exception("[%s/%s] Failed %s: %s", i, total, name, exc)
                results[name] = {"status": "error", "error": str(exc)}

        return results

    def get_stats(self, output_dir: str) -> Dict[str, Any]:
        output_path = Path(output_dir)
        if not output_path.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        dataset_files = sorted(p for p in output_path.glob("*.jsonl") if p.is_file())
        stats: Dict[str, Any] = {"datasets": {}, "totals": {"lines": 0, "bytes": 0, "estimated_tokens": 0}}
        lang_counter = Counter()

        token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

        for file_path in dataset_files:
            line_count = 0
            estimated_tokens = 0
            file_bytes = file_path.stat().st_size
            file_lang_counter = Counter()

            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line_count += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = obj.get("text", "")
                    estimated_tokens += len(token_pattern.findall(text)) if isinstance(text, str) else 0
                    language = obj.get("language", "unknown")
                    file_lang_counter[language] += 1
                    lang_counter[language] += 1

            stats["datasets"][file_path.stem] = {
                "file": str(file_path),
                "lines": line_count,
                "bytes": file_bytes,
                "estimated_tokens": estimated_tokens,
                "language_distribution": dict(file_lang_counter),
            }
            stats["totals"]["lines"] += line_count
            stats["totals"]["bytes"] += file_bytes
            stats["totals"]["estimated_tokens"] += estimated_tokens

        stats["totals"]["language_distribution"] = dict(lang_counter)
        return stats
