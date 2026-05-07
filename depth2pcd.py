import numpy as np
from PIL import Image
from pypcd_imp import pypcd
import argparse
import os


def load_depth_data(npz_path, argument_focal_length=None):
    data = np.load(npz_path)
    depth = data["depth"]
    if "focallength_px" not in data.files:
        if argument_focal_length:
            focal = float(argument_focal_length)
        else:
            raise ValueError("Unknown focal length for depth image unprojection. Neither in .npz file, nor given as command line argument.")
    else:
        focal = data["focallength_px"]
    return depth, focal


def load_rgb_image(image_path, target_shape):
    if image_path is None:
        return None
    image = Image.open(image_path).convert("RGB").resize((target_shape[1], target_shape[0]))
    return np.asarray(image)


def load_nir_image(image_path, target_shape):
    """Load a single-channel NIR image and resize to depth resolution.

    Uses bilinear resampling since NIR is a continuous intensity signal,
    not categorical like a label map.
    """
    if image_path is None:
        return None
    image = Image.open(image_path).convert("L").resize(
        (target_shape[1], target_shape[0]), Image.BILINEAR
    )
    return np.asarray(image)


def load_label_image(label_path, target_shape):
    if label_path is None:
        return None
    # NEAREST resampling is critical for label maps — never interpolate class IDs
    image = Image.open(label_path).resize((target_shape[1], target_shape[0]), Image.NEAREST)
    labels = np.asarray(image)
    # Some label PNGs are saved as RGB or RGBA; class IDs are typically in a single channel
    if labels.ndim == 3:
        labels = labels[..., 0]
    return labels


def dilate_mask(mask, dilate_pixels):
    """Grow a boolean mask by `dilate_pixels` in all 8 directions (Chebyshev)."""
    if dilate_pixels <= 0:
        return mask
    out = mask.copy()
    for _ in range(dilate_pixels):
        shifted = out.copy()
        shifted[:, :-1] |= out[:, 1:]
        shifted[:, 1:]  |= out[:, :-1]
        shifted[:-1, :] |= out[1:, :]
        shifted[1:, :]  |= out[:-1, :]
        out = shifted
    return out


def compute_boundary_mask(labels, dilate_pixels):
    """Return a boolean mask (True = within `dilate_pixels` of a class boundary)."""
    if dilate_pixels <= 0:
        return np.zeros_like(labels, dtype=bool)

    # A pixel is on a boundary if any 4-neighbor has a different label.
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]   # right neighbor
    boundary[:, 1:]  |= labels[:, 1:]  != labels[:, :-1]  # left neighbor
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]   # bottom neighbor
    boundary[1:, :]  |= labels[1:, :]  != labels[:-1, :]  # top neighbor

    return dilate_mask(boundary, dilate_pixels)


def compute_depth_discontinuity_mask(depth, rel_threshold, dilate_pixels):
    """Return a boolean mask (True = within `dilate_pixels` of a depth discontinuity).

    A pixel is flagged as a discontinuity if its depth differs from a neighbor
    by more than `rel_threshold` times its own depth — i.e. a relative jump.
    This catches occlusion boundaries (foreground/background transitions and
    object/sky transitions) where monocular depth networks tend to produce
    flying-pixel ramps.
    """
    # Absolute depth jumps to right and bottom neighbors
    dx = np.zeros_like(depth)
    dy = np.zeros_like(depth)
    dx[:, :-1] = np.abs(depth[:, 1:] - depth[:, :-1])
    dy[:-1, :] = np.abs(depth[1:, :] - depth[:-1, :])
    grad = np.maximum(dx, dy)

    # Relative threshold scales with depth: a 5% jump on a 10m point = 0.5m,
    # but on a 100m point = 5m. This matches how depth uncertainty actually scales.
    # Guard against zero/negative depth to avoid divide-by-zero.
    safe_depth = np.maximum(depth, 1e-6)
    discontinuity = grad > (rel_threshold * safe_depth)

    return dilate_mask(discontinuity, dilate_pixels)


def depth_to_pcd_struct(depth, focal_length, rgb=None, nir=None, labels=None,
                        exclude_ids=None, boundary_erode=0, min_distance=0.0,
                        depth_boundary_erode=0, depth_rel_threshold=0.05):
    h, w = depth.shape
    cx, cy = w / 2.0, h / 2.0
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x = (xx - cx) * depth / focal_length
    y = (yy - cy) * depth / focal_length
    z = depth
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    keep = np.ones(points.shape[0], dtype=bool)

    if labels is not None and exclude_ids:
        labels_flat = labels.reshape(-1)
        excluded = np.isin(labels_flat, np.asarray(list(exclude_ids), dtype=labels_flat.dtype))
        keep &= ~excluded
        print(f"Excluding {excluded.sum()} points based on labels {sorted(exclude_ids)}")

    if labels is not None and boundary_erode > 0:
        boundary_mask = compute_boundary_mask(labels, boundary_erode)
        boundary_flat = boundary_mask.reshape(-1)
        keep &= ~boundary_flat
        print(f"Excluding {boundary_flat.sum()} points within {boundary_erode}px of a semantic class boundary")

    if depth_boundary_erode > 0:
        depth_disc_mask = compute_depth_discontinuity_mask(
            depth, depth_rel_threshold, depth_boundary_erode
        )
        depth_disc_flat = depth_disc_mask.reshape(-1)
        keep &= ~depth_disc_flat
        print(f"Excluding {depth_disc_flat.sum()} points within {depth_boundary_erode}px "
              f"of a depth discontinuity (rel_threshold={depth_rel_threshold})")

    if min_distance > 0.0:
        # Euclidean distance from origin, squared (avoids a sqrt per point).
        dist_sq = (points ** 2).sum(axis=1)
        too_close = dist_sq < (min_distance ** 2)
        keep &= ~too_close
        print(f"Excluding {too_close.sum()} points closer than {min_distance}m from origin")

    print(f"Keeping {keep.sum()} of {keep.size} points")
    points = points[keep]

    # Build the PCD record. RGB and NIR are mutually exclusive: pick the schema
    # that matches what was provided.
    if nir is not None:
        nir_flat = nir.reshape(-1)[keep].astype(np.uint32)

        pc_data = np.zeros(points.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'u4')
        ])
        pc_data['x'] = points[:, 0]
        pc_data['y'] = points[:, 1]
        pc_data['z'] = points[:, 2]
        pc_data['intensity'] = nir_flat

        pc_header = {
            'version': .7,
            'fields': ['x', 'y', 'z', 'intensity'],
            'size': [4, 4, 4, 4],
            'type': ['F', 'F', 'F', 'U'],   # U = unsigned int
            'count': [1, 1, 1, 1],
            'width': pc_data.shape[0],
            'height': 1,
            'viewpoint': [0, 0, 0, 1, 0, 0, 0],
            'points': pc_data.shape[0],
            'data': 'binary',
        }
    else:
        if rgb is not None:
            rgb_flat = rgb.reshape(-1, 3)[keep]
            rgb_uint32 = (
                rgb_flat[:, 0].astype(np.uint32) << 16 |
                rgb_flat[:, 1].astype(np.uint32) << 8 |
                rgb_flat[:, 2].astype(np.uint32)
            )
            rgb_packed = rgb_uint32.view(np.float32)
        else:
            rgb_packed = np.zeros(points.shape[0], dtype=np.float32)

        pc_data = np.zeros(points.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')
        ])
        pc_data['x'] = points[:, 0]
        pc_data['y'] = points[:, 1]
        pc_data['z'] = points[:, 2]
        pc_data['rgb'] = rgb_packed

        pc_header = {
            'version': .7,
            'fields': ['x', 'y', 'z', 'rgb'],
            'size': [4, 4, 4, 4],
            'type': ['F', 'F', 'F', 'F'],
            'count': [1, 1, 1, 1],
            'width': pc_data.shape[0],
            'height': 1,
            'viewpoint': [0, 0, 0, 1, 0, 0, 0],
            'points': pc_data.shape[0],
            'data': 'binary',
        }

    return pypcd.PointCloud(pc_header, pc_data)


def parse_id_list(s):
    if s is None:
        return None
    return [int(x) for x in s.split(',') if x.strip()]


def main(npz_path, image_path=None, nir_path=None, focal_cli_argument=None,
         output_pcd="output.pcd", label_path=None, exclude_ids=None,
         boundary_erode=0, min_distance=0.0,
         depth_boundary_erode=0, depth_rel_threshold=0.05):
    if image_path is not None and nir_path is not None:
        raise ValueError("Use either --image (RGB) or --nir, not both.")

    depth, focal = load_depth_data(npz_path, focal_cli_argument)
    rgb = load_rgb_image(image_path, depth.shape) if image_path else None
    nir = load_nir_image(nir_path, depth.shape) if nir_path else None
    labels = load_label_image(label_path, depth.shape) if label_path else None

    if (exclude_ids or boundary_erode > 0) and labels is None:
        raise ValueError("--exclude-labels or --erode-semantic-boundary requires --labels.")

    pcd = depth_to_pcd_struct(depth, focal, rgb=rgb, nir=nir, labels=labels,
                              exclude_ids=exclude_ids, boundary_erode=boundary_erode,
                              min_distance=min_distance,
                              depth_boundary_erode=depth_boundary_erode,
                              depth_rel_threshold=depth_rel_threshold)
    pcd.save(output_pcd)
    print(f"Saved PCD file: {output_pcd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DepthPro .npz to PCD using pypcd-imp")
    parser.add_argument("npz_path", help="Path to the .npz depth file")
    parser.add_argument("--image", help="Optional RGB image to colorize point cloud", default=None)
    parser.add_argument("--nir", help="Optional grayscale NIR image; stored as 'intensity' field "
                                      "instead of packed RGB. Resized to depth resolution.",
                        default=None)
    parser.add_argument("--focal", help="Optional focal length if not given in .npz depth file", default=None)
    parser.add_argument("--output", help="Output PCD filename", default="output.pcd")
    parser.add_argument("--labels", help="Optional semantic label PNG (e.g. *_labelids.png)", default=None)
    parser.add_argument("--exclude-labels",
                        help="Comma-separated list of class IDs to exclude, e.g. '0,1,23'",
                        default=None)
    parser.add_argument("--erode-semantic-boundary", type=int, default=0,
                        help="Remove points within N pixels of any semantic class boundary")
    parser.add_argument("--min-distance", type=float, default=0.0,
                        help="Remove points closer than N meters from the camera origin")
    parser.add_argument("--erode-depth-boundary", type=int, default=0,
                        help="Remove points within N pixels of a depth discontinuity "
                             "(removes 'flying pixels' at object/background and object/sky borders)")
    parser.add_argument("--depth-threshold", type=float, default=0.05,
                        help="Relative depth jump that counts as a discontinuity "
                             "(default 0.05 = 5%% of local depth)")
    args = parser.parse_args()

    exclude_ids = parse_id_list(args.exclude_labels)
    main(args.npz_path, image_path=args.image, nir_path=args.nir,
         focal_cli_argument=args.focal, output_pcd=args.output,
         label_path=args.labels, exclude_ids=exclude_ids,
         boundary_erode=args.erode_semantic_boundary,
         min_distance=args.min_distance,
         depth_boundary_erode=args.erode_depth_boundary,
         depth_rel_threshold=args.depth_threshold)