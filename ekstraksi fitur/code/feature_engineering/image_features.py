from __future__ import annotations

import io
import os
import time
import urllib.request
from urllib.parse import quote, urlsplit, urlunsplit
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image

from .specification import FeatureSpec, get_default_feature_specs

try:
    import torch
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    ResNet18_Weights = None  # type: ignore[assignment]
    FasterRCNN_ResNet50_FPN_Weights = None  # type: ignore[assignment]
    FCN_ResNet50_Weights = None  # type: ignore[assignment]
    resnet18 = None  # type: ignore[assignment]
    fasterrcnn_resnet50_fpn = None  # type: ignore[assignment]
    fcn_resnet50 = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


ExtractorFunction = Callable[[np.ndarray, Dict[str, Any], Dict[str, Any]], Any]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _to_float_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Image array must have shape (H, W, 3).")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _rgb_to_hsv_normalized(image_rgb: np.ndarray) -> np.ndarray:
    rgb = _to_float_array(image_rgb)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nonzero = delta > 1e-12
    s[nonzero] = delta[nonzero] / np.maximum(maxc[nonzero], 1e-12)

    r_mask = nonzero & (maxc == r)
    g_mask = nonzero & (maxc == g)
    b_mask = nonzero & (maxc == b)

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    h = h / 6.0

    return np.stack([h, s, v], axis=-1)


class ImageFeatureExtractor:
    def __init__(
        self,
        specs: Sequence[FeatureSpec] | None = None,
        image_root: str | None = None,
        timeout_sec: int = 10,
    ):
        all_specs = list(specs) if specs is not None else get_default_feature_specs()
        self.specs = [s for s in all_specs if s.modality == "image" and s.enabled]
        self.image_root = Path(image_root) if image_root else None
        self.timeout_sec = timeout_sec
        self.backend = os.environ.get("AES_IMAGE_BACKEND", "deep").strip().lower()
        self.use_deep_backend = bool(_TORCH_AVAILABLE and self.backend != "proxy")
        self.device = "cpu"
        self._feature_cache: Dict[str, Dict[str, float]] = {}
        self._box_model: Any = None
        self._point_model: Any = None
        self._legend_backbone: Any = None
        self._legend_projection: Any = None
        self._imagenet_mean: Any = None
        self._imagenet_std: Any = None
        self.registry: Dict[str, ExtractorFunction] = {
            "box_detector_cascade_resnet50_fpn": self._box_detector_cascade_resnet50_fpn,
            "point_detector_fcn_fusion": self._point_detector_fcn_fusion,
            "legend_matching_embedding": self._legend_matching_embedding,
        }

    def expected_feature_columns(self) -> List[str]:
        cols: List[str] = []
        for spec in self.specs:
            cols.extend(spec.output_columns)
        return cols

    def extract(self, row: Mapping[str, Any]) -> Dict[str, float]:
        graph_value = str(row.get("graph", "") or "").strip()
        if graph_value in self._feature_cache:
            return dict(self._feature_cache[graph_value])

        features: Dict[str, float] = {}
        try:
            image_rgb, meta = self._load_image_rgb(graph_value)

            for spec in self.specs:
                fallback_map = {col: spec.fallback_value for col in spec.output_columns}
                func = self.registry.get(spec.extractor_key)
                if func is None:
                    features.update(fallback_map)
                    continue

                try:
                    raw_result = func(image_rgb, meta, spec.params)
                    mapped = self._map_result(raw_result, spec)
                    for col in spec.output_columns:
                        features[col] = float(mapped.get(col, spec.fallback_value))
                except Exception:
                    features.update(fallback_map)

            self._feature_cache[graph_value] = dict(features)
            return features
        except Exception:
            for spec in self.specs:
                for col in spec.output_columns:
                    features[col] = spec.fallback_value
            return features

    @staticmethod
    def _map_result(raw_result: Any, spec: FeatureSpec) -> Dict[str, float]:
        if isinstance(raw_result, dict):
            mapped: Dict[str, float] = {}
            for col in spec.output_columns:
                if col in raw_result:
                    mapped[col] = float(raw_result[col])
                    continue
                key_without_prefix = col.replace("img_", "", 1)
                if key_without_prefix in raw_result:
                    mapped[col] = float(raw_result[key_without_prefix])
                    continue
                mapped[col] = spec.fallback_value
            return mapped

        if isinstance(raw_result, (list, tuple, np.ndarray)):
            arr = list(raw_result)
            return {
                col: float(arr[idx]) if idx < len(arr) else spec.fallback_value
                for idx, col in enumerate(spec.output_columns)
            }

        if not spec.output_columns:
            return {}
        return {spec.output_columns[0]: float(raw_result)}

    def _load_image_rgb(self, graph_value: str) -> tuple[np.ndarray, Dict[str, Any]]:
        if not graph_value:
            raise ValueError("graph value is empty")

        is_url = graph_value.lower().startswith(("http://", "https://"))

        if is_url:
            candidate_urls = self._build_url_candidates(graph_value)
            last_error: Exception | None = None
            binary = None
            used_url = graph_value

            for candidate in candidate_urls:
                for attempt in range(3):
                    try:
                        request = urllib.request.Request(
                            candidate,
                            headers={"User-Agent": "Mozilla/5.0 (AES-Feature-Extractor)"},
                        )
                        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                            binary = response.read()
                        used_url = candidate
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < 2:
                            time.sleep(0.35 * (attempt + 1))
                if binary is not None:
                    break

            if binary is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError(f"Unable to read image URL: {graph_value}")

            img = Image.open(io.BytesIO(binary)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            h, w = arr.shape[:2]
            meta = {
                "is_url": True,
                "source_url": used_url,
                "width": float(w),
                "height": float(h),
                "area": float(w * h),
            }
            return arr, meta

        path = Path(graph_value)
        if not path.is_absolute() and self.image_root is not None:
            path = (self.image_root / path).resolve()

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
        meta = {
            "is_url": False,
            "width": float(w),
            "height": float(h),
            "area": float(w * h),
            "file_size_kb": _safe_div(float(os.path.getsize(path)), 1024.0),
        }
        return arr, meta

    @staticmethod
    def _build_url_candidates(url: str) -> List[str]:
        candidates: List[str] = [url]

        try:
            parsed = urlsplit(url)
            requoted_path = quote(parsed.path, safe="/%")
            normalized = urlunsplit((parsed.scheme, parsed.netloc, requoted_path, parsed.query, parsed.fragment))
            if normalized not in candidates:
                candidates.append(normalized)

            # Fallback for known dataset mirror naming differences.
            if "raw.githubusercontent.com" in parsed.netloc and "/jsu360/MLLM-for-AES/" in requoted_path:
                mirror = normalized.replace("/jsu360/MLLM-for-AES/", "/jsu360/MLLM-for-AES-graph/")
                if mirror not in candidates:
                    candidates.append(mirror)
            elif "raw.githubusercontent.com" in parsed.netloc and "/jsu360/MLLM-for-AES-graph/" in requoted_path:
                mirror = normalized.replace("/jsu360/MLLM-for-AES-graph/", "/jsu360/MLLM-for-AES/")
                if mirror not in candidates:
                    candidates.append(mirror)
        except Exception:
            pass

        return candidates

    def _ensure_imagenet_stats(self) -> None:
        if not self.use_deep_backend:
            return
        if self._imagenet_mean is not None and self._imagenet_std is not None:
            return
        self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _to_torch_image(self, image_rgb: np.ndarray) -> Any:
        rgb = _to_float_array(image_rgb)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()

    def _normalize_imagenet(self, image_chw: Any) -> Any:
        self._ensure_imagenet_stats()
        return (image_chw - self._imagenet_mean) / self._imagenet_std

    def _resize_if_large(self, image_chw: Any, max_side: int = 800) -> Any:
        _, h, w = image_chw.shape
        side = max(h, w)
        if side <= max_side:
            return image_chw
        scale = max_side / float(side)
        nh = max(int(h * scale), 32)
        nw = max(int(w * scale), 32)
        return F.interpolate(image_chw.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)

    def _ensure_box_model(self) -> None:
        if not self.use_deep_backend:
            raise RuntimeError("Deep backend unavailable for box detector")
        if self._box_model is not None:
            return
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        self._box_model = model

    def _ensure_point_model(self) -> None:
        if not self.use_deep_backend:
            raise RuntimeError("Deep backend unavailable for point detector")
        if self._point_model is not None:
            return
        weights = FCN_ResNet50_Weights.DEFAULT
        model = fcn_resnet50(weights=weights)
        model.eval()
        self._point_model = model

    def _ensure_legend_model(self) -> None:
        if not self.use_deep_backend:
            raise RuntimeError("Deep backend unavailable for embedding model")
        if self._legend_backbone is not None and self._legend_projection is not None:
            return

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self._legend_backbone = model

        gen = torch.Generator().manual_seed(42)
        projection = torch.randn(512, 128, generator=gen, dtype=torch.float32)
        # L2 normalize columns for stable embedding projection.
        projection = projection / (torch.norm(projection, dim=0, keepdim=True) + 1e-8)
        self._legend_projection = projection

    def _embed_patch_deep(self, patch_rgb: np.ndarray) -> np.ndarray:
        self._ensure_legend_model()

        patch_chw = self._to_torch_image(patch_rgb)
        patch_chw = F.interpolate(patch_chw.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        patch_chw = self._normalize_imagenet(patch_chw)

        with torch.no_grad():
            feat = self._legend_backbone(patch_chw.unsqueeze(0)).squeeze(0)
            emb = torch.matmul(feat, self._legend_projection)
            emb = emb / (torch.norm(emb) + 1e-8)

        return emb.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _to_gray(image_rgb: np.ndarray) -> np.ndarray:
        rgb = _to_float_array(image_rgb)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _downsample2(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            return arr
        return arr[::2, ::2]

    @staticmethod
    def _gaussian_blur3(gray: np.ndarray) -> np.ndarray:
        # Lightweight Gaussian blur to emulate heatmap smoothing.
        kernel = np.array(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=np.float32,
        ) / 16.0
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        out = np.zeros_like(gray, dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += kernel[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
        return out

    @staticmethod
    def _count_connected_components(binary_mask: np.ndarray, min_size: int = 3) -> int:
        h, w = binary_mask.shape
        visited = np.zeros((h, w), dtype=bool)
        count = 0

        for y in range(h):
            for x in range(w):
                if not binary_mask[y, x] or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True
                size = 0

                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    y0 = max(cy - 1, 0)
                    y1 = min(cy + 2, h)
                    x0 = max(cx - 1, 0)
                    x1 = min(cx + 2, w)
                    for ny in range(y0, y1):
                        for nx in range(x0, x1):
                            if binary_mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))

                if size >= min_size:
                    count += 1

        return count

    @staticmethod
    def _pseudo_embedding_128(patch_rgb: np.ndarray) -> np.ndarray:
        rgb = _to_float_array(patch_rgb)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

        # 32 bins per channel => 96 dims
        r_hist, _ = np.histogram(rgb[..., 0], bins=32, range=(0.0, 1.0), density=True)
        g_hist, _ = np.histogram(rgb[..., 1], bins=32, range=(0.0, 1.0), density=True)
        b_hist, _ = np.histogram(rgb[..., 2], bins=32, range=(0.0, 1.0), density=True)

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
        gy[1:, :] = gray[1:, :] - gray[:-1, :]
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.mod(np.arctan2(gy, gx), 2.0 * np.pi)

        # 32 dims orientation-weighted histogram => total 128 dims
        ori_hist, _ = np.histogram(ang, bins=32, range=(0.0, 2.0 * np.pi), weights=mag, density=True)

        emb = np.concatenate([r_hist, g_hist, b_hist, ori_hist], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-12:
            emb = emb / norm
        return emb

    def _box_detector_cascade_resnet50_fpn(
        self, image_rgb: np.ndarray, _: Dict[str, Any], __: Dict[str, Any]
    ) -> float:
        if self.use_deep_backend:
            try:
                return self._box_detector_cascade_resnet50_fpn_deep(image_rgb)
            except Exception:
                pass
        return self._box_detector_cascade_resnet50_fpn_proxy(image_rgb)

    def _box_detector_cascade_resnet50_fpn_deep(self, image_rgb: np.ndarray) -> float:
        self._ensure_box_model()
        image_chw = self._resize_if_large(self._to_torch_image(image_rgb), max_side=1024)

        with torch.no_grad():
            pred = self._box_model([image_chw])[0]

        if "scores" not in pred or pred["scores"].numel() == 0:
            return 0.0

        scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
        top = np.sort(scores)[::-1][:20]

        stage_05 = float(np.mean(top >= 0.50)) if top.size else 0.0
        stage_06 = float(np.mean(top >= 0.60)) if top.size else 0.0
        stage_07 = float(np.mean(top >= 0.70)) if top.size else 0.0
        mean_top = float(np.mean(top)) if top.size else 0.0

        # Cascade-style stricter stages are approximated by progressive score gates.
        score = 0.45 * mean_top + 0.20 * stage_05 + 0.20 * stage_06 + 0.15 * stage_07
        return self._clamp01(score)

    def _box_detector_cascade_resnet50_fpn_proxy(self, image_rgb: np.ndarray) -> float:
        gray = self._to_gray(image_rgb)
        scales = [gray, self._downsample2(gray), self._downsample2(self._downsample2(gray))]

        edge_densities: List[float] = []
        hv_scores: List[float] = []
        compactness_scores: List[float] = []

        for level in scales:
            if level.size == 0:
                continue

            gx = np.zeros_like(level)
            gy = np.zeros_like(level)
            gx[:, 1:] = level[:, 1:] - level[:, :-1]
            gy[1:, :] = level[1:, :] - level[:-1, :]
            mag = np.sqrt(gx * gx + gy * gy)
            ang = np.mod(np.arctan2(gy, gx), np.pi)

            strong = mag > np.percentile(mag, 90)
            edge_density = float(np.mean(strong))
            edge_densities.append(edge_density)

            hv = np.logical_or(np.abs(ang - 0.0) < 0.25, np.abs(ang - np.pi / 2.0) < 0.25)
            hv_score = _safe_div(float(np.sum(hv & strong)), float(np.sum(strong)))
            hv_scores.append(hv_score)

            compact = 1.0 - min(abs(edge_density - 0.12) / 0.12, 1.0)
            compactness_scores.append(compact)

        if not edge_densities:
            return 0.0

        edge_mean = float(np.mean(edge_densities))
        scale_consistency = 1.0 - _safe_div(float(np.std(edge_densities)), edge_mean + 1e-6)
        scale_consistency = self._clamp01(scale_consistency)
        hv_mean = self._clamp01(float(np.mean(hv_scores)) if hv_scores else 0.0)
        compact_mean = self._clamp01(float(np.mean(compactness_scores)) if compactness_scores else 0.0)

        score = 0.40 * scale_consistency + 0.35 * hv_mean + 0.25 * compact_mean
        return self._clamp01(score)

    def _point_detector_fcn_fusion(
        self, image_rgb: np.ndarray, _: Dict[str, Any], __: Dict[str, Any]
    ) -> float:
        if self.use_deep_backend:
            try:
                return self._point_detector_fcn_fusion_deep(image_rgb)
            except Exception:
                pass
        return self._point_detector_fcn_fusion_proxy(image_rgb)

    def _point_detector_fcn_fusion_deep(self, image_rgb: np.ndarray) -> float:
        self._ensure_point_model()

        image_chw = self._to_torch_image(image_rgb)
        _, h0, w0 = image_chw.shape
        resized = self._resize_if_large(image_chw, max_side=800)
        normalized = self._normalize_imagenet(resized)

        with torch.no_grad():
            logits = self._point_model(normalized.unsqueeze(0))["out"][0]
            heat = torch.softmax(logits, dim=0).max(dim=0).values

        if heat.shape != (h0, w0):
            heat = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False)
            heat = heat.squeeze(0).squeeze(0)

        heat_np = heat.detach().cpu().numpy().astype(np.float32)

        h, w = heat_np.shape
        y0 = int(0.10 * h)
        y1 = int(0.90 * h)
        x0 = int(0.10 * w)
        x1 = int(0.90 * w)
        core = heat_np[y0:y1, x0:x1]
        if core.size == 0:
            return 0.0

        threshold = float(np.percentile(core, 93))
        mask = core >= threshold

        up = np.pad(core[1:, :], ((0, 1), (0, 0)), mode="edge")
        down = np.pad(core[:-1, :], ((1, 0), (0, 0)), mode="edge")
        left = np.pad(core[:, 1:], ((0, 0), (0, 1)), mode="edge")
        right = np.pad(core[:, :-1], ((0, 0), (1, 0)), mode="edge")
        local_max = (core >= up) & (core >= down) & (core >= left) & (core >= right) & mask

        mask_ds = mask[::2, ::2]
        comps = self._count_connected_components(mask_ds, min_size=2)
        peaks = int(np.sum(local_max))

        separability = _safe_div(float(comps), float(peaks) + 1e-6)
        confidence = self._clamp01(_safe_div(float(np.sum(mask)), mask.size))
        peak_density = self._clamp01(_safe_div(float(peaks), float(mask.size) / 250.0 + 1e-6))

        score = 0.45 * self._clamp01(separability) + 0.30 * confidence + 0.25 * peak_density
        return self._clamp01(score)

    def _point_detector_fcn_fusion_proxy(self, image_rgb: np.ndarray) -> float:
        gray = self._to_gray(image_rgb)
        heat = self._gaussian_blur3(gray)

        h, w = heat.shape
        y0 = int(0.10 * h)
        y1 = int(0.90 * h)
        x0 = int(0.10 * w)
        x1 = int(0.90 * w)
        core = heat[y0:y1, x0:x1]
        if core.size == 0:
            return 0.0

        threshold = float(np.percentile(core, 93))
        mask = core >= threshold

        up = np.pad(core[1:, :], ((0, 1), (0, 0)), mode="edge")
        down = np.pad(core[:-1, :], ((1, 0), (0, 0)), mode="edge")
        left = np.pad(core[:, 1:], ((0, 0), (0, 1)), mode="edge")
        right = np.pad(core[:, :-1], ((0, 0), (1, 0)), mode="edge")
        local_max = (core >= up) & (core >= down) & (core >= left) & (core >= right) & mask

        mask_ds = mask[::2, ::2]
        comps = self._count_connected_components(mask_ds, min_size=2)
        peaks = int(np.sum(local_max))

        separability = _safe_div(float(comps), float(peaks) + 1e-6)
        confidence = self._clamp01(_safe_div(float(np.sum(mask)), mask.size))
        peak_density = self._clamp01(_safe_div(float(peaks), float(mask.size) / 250.0 + 1e-6))

        score = 0.45 * self._clamp01(separability) + 0.30 * confidence + 0.25 * peak_density
        return self._clamp01(score)

    def _legend_matching_embedding(
        self, image_rgb: np.ndarray, _: Dict[str, Any], __: Dict[str, Any]
    ) -> float:
        if self.use_deep_backend:
            try:
                return self._legend_matching_embedding_deep(image_rgb)
            except Exception:
                pass
        return self._legend_matching_embedding_proxy(image_rgb)

    def _legend_matching_embedding_deep(self, image_rgb: np.ndarray) -> float:
        rgb = _to_float_array(image_rgb)
        h, w, _ = rgb.shape
        if h < 8 or w < 8:
            return 0.0

        cy0, cy1 = int(0.2 * h), int(0.8 * h)
        cx0, cx1 = int(0.2 * w), int(0.8 * w)
        plot_patch = rgb[cy0:cy1, cx0:cx1, :]

        margin = int(0.2 * min(h, w))
        margin = max(margin, 4)
        candidates = [
            rgb[:margin, :, :],
            rgb[h - margin :, :, :],
            rgb[:, :margin, :],
            rgb[:, w - margin :, :],
        ]

        plot_emb = self._embed_patch_deep(plot_patch)
        dists: List[float] = []
        for patch in candidates:
            emb = self._embed_patch_deep(patch)
            dists.append(float(np.linalg.norm(plot_emb - emb)))

        if not dists:
            return 0.0

        min_dist = min(dists)
        similarity = 1.0 / (1.0 + min_dist)
        return self._clamp01(similarity)

    def _legend_matching_embedding_proxy(self, image_rgb: np.ndarray) -> float:
        rgb = _to_float_array(image_rgb)
        h, w, _ = rgb.shape
        if h < 8 or w < 8:
            return 0.0

        cy0, cy1 = int(0.2 * h), int(0.8 * h)
        cx0, cx1 = int(0.2 * w), int(0.8 * w)
        plot_patch = rgb[cy0:cy1, cx0:cx1, :]

        margin = int(0.2 * min(h, w))
        margin = max(margin, 4)
        candidates = [
            rgb[:margin, :, :],
            rgb[h - margin :, :, :],
            rgb[:, :margin, :],
            rgb[:, w - margin :, :],
        ]

        plot_emb = self._pseudo_embedding_128(plot_patch)
        dists: List[float] = []
        for patch in candidates:
            emb = self._pseudo_embedding_128(patch)
            dists.append(float(np.linalg.norm(plot_emb - emb)))

        if not dists:
            return 0.0

        min_dist = min(dists)
        similarity = 1.0 / (1.0 + min_dist)
        return self._clamp01(similarity)
