import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def configure_chinese_font():
    preferred_fonts = [
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans TC",
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    available_fonts = [
        font_name for font_name in preferred_fonts if font_name in installed_fonts
    ]
    plt.rcParams["font.sans-serif"] = available_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


configure_chinese_font()


REQUIRED_COLUMNS = [
    "箱型名稱",
    "長度(cm)",
    "寬度(cm)",
    "高度(cm)",
    "數量(箱)",
    "單箱重量(kg)",
    "顏色代碼(HEX)",
]
ROTATION_COLUMN = "可平面旋轉"
TIPPING_COLUMN = "可翻面"
LEGACY_ROTATION_COLUMN = "可轉向"
PACK_COLUMN = "排入棧板"


def parse_bool(value, default=True):
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "是", "可", "可以"}:
        return True
    if text in {"false", "f", "no", "n", "0", "否", "不可", "不可以"}:
        return False
    return default


def can_fit_on_pallet(
    length,
    width,
    height,
    pallet_length,
    pallet_width,
    can_rotate=True,
    can_tip=False,
):
    base_orientations = [(length, width)]
    if can_rotate:
        base_orientations.append((width, length))
    if can_tip:
        base_orientations.extend(
            [(length, height), (height, length), (width, height), (height, width)]
        )
    return any(
        box_w <= pallet_length and box_h <= pallet_width
        for box_w, box_h in base_orientations
    )


def normalize_cargo(
    cargo_df,
    pallet_length,
    pallet_width,
    default_can_rotate=True,
    default_can_tip=False,
):
    cargo_items = []
    invalid_rows = []
    oversized_items = []

    for row_idx, row in cargo_df.iterrows():
        if not parse_bool(row.get(PACK_COLUMN, True), default=True):
            continue
        try:
            name = str(row["箱型名稱"]).strip()
            length = float(row["長度(cm)"])
            width = float(row["寬度(cm)"])
            height = float(row["高度(cm)"])
            count = int(row["數量(箱)"])
            weight = float(row["單箱重量(kg)"])
            color = str(row["顏色代碼(HEX)"]).strip() or "#CCCCCC"
            can_rotate = parse_bool(
                row.get(
                    ROTATION_COLUMN,
                    row.get(LEGACY_ROTATION_COLUMN, default_can_rotate),
                ),
                default=default_can_rotate,
            )
            can_tip = parse_bool(
                row.get(TIPPING_COLUMN, default_can_tip),
                default=default_can_tip,
            )

            if not name or min(length, width, height, count, weight) <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            invalid_rows.append(row_idx + 1)
            continue

        for box_no in range(1, count + 1):
            item = {
                "id": f"{row_idx + 1}-{box_no}",
                "name": name,
                "length": length,
                "width": width,
                "height": height,
                "weight": weight,
                "color": color,
                "area": length * width,
                "volume": length * width * height,
                "box_count": 1,
                "can_rotate": can_rotate,
                "can_tip": can_tip,
            }
            if can_fit_on_pallet(
                length,
                width,
                height,
                pallet_length,
                pallet_width,
                can_rotate,
                can_tip,
            ):
                cargo_items.append(item)
            else:
                oversized_items.append(item)

    cargo_items.sort(key=foundation_priority)
    return cargo_items, invalid_rows, oversized_items


def foundation_priority(item):
    return (
        -foundation_score(item),
        -item["volume"],
        -item["weight"],
        -max_possible_base_area(item),
        -max(item["length"], item["width"], item["height"]),
        item["name"],
    )


def gap_fill_priority(item):
    return (
        foundation_score(item),
        item["volume"],
        max_possible_base_area(item),
        item["weight"],
        item["height"],
        item["name"],
    )


def dimension_stats(item):
    dims = sorted([item["length"], item["width"], item["height"]])
    min_dim, mid_dim, max_dim = dims
    slenderness = min_dim / max_dim if max_dim > 0 else 1
    flatness = min_dim / mid_dim if mid_dim > 0 else 1
    return min_dim, mid_dim, max_dim, slenderness, flatness


def max_possible_base_area(item):
    dims = [item["length"], item["width"], item["height"]]
    return max(dims[0] * dims[1], dims[0] * dims[2], dims[1] * dims[2])


def foundation_score(item):
    _, _, _, slenderness, flatness = dimension_stats(item)
    shape_penalty = 0.35 if is_gap_fill_item(item) else 1.0
    return (
        item["volume"]
        * max(item["weight"], 0.1)
        * max_possible_base_area(item)
        * slenderness
        * flatness
        * shape_penalty
    )


def is_gap_fill_item(item):
    _, _, _, slenderness, flatness = dimension_stats(item)
    return slenderness <= 0.28 or flatness <= 0.45


def sort_for_humanized_placement(items, mode="foundation"):
    key_func = foundation_priority if mode == "foundation" else gap_fill_priority
    return sorted(
        items,
        key=key_func,
    )


def get_orientations(item, pallet_length, pallet_width):
    orientations = [(item["length"], item["width"], item["height"])]
    if item.get("can_rotate", True):
        orientations.append((item["width"], item["length"], item["height"]))
    if item.get("can_tip", False):
        for orientation in (
            (item["length"], item["height"], item["width"]),
            (item["width"], item["height"], item["length"]),
            (item["height"], item["length"], item["width"]),
            (item["height"], item["width"], item["length"]),
        ):
            if orientation not in orientations:
                orientations.append(orientation)

    return [
        (float(box_w), float(box_h), float(box_z))
        for box_w, box_h, box_z in orientations
        if box_w <= pallet_length and box_h <= pallet_width
    ]


def get_orientation_label(item, box_w, box_h, box_z):
    if (
        math.isclose(box_w, item["length"])
        and math.isclose(box_h, item["width"])
        and math.isclose(box_z, item["height"])
    ):
        return "原方向"
    if math.isclose(box_z, item["height"]):
        return "平面轉向90度"
    return "立放/側放"


def overlap_area(item_a, item_b):
    x_overlap = max(
        0,
        min(item_a["x"] + item_a["box_w"], item_b["x"] + item_b["box_w"])
        - max(item_a["x"], item_b["x"]),
    )
    y_overlap = max(
        0,
        min(item_a["y"] + item_a["box_h"], item_b["y"] + item_b["box_h"])
        - max(item_a["y"], item_b["y"]),
    )
    return x_overlap * y_overlap


def boxes_overlap_3d(item_a, item_b):
    if overlap_area(item_a, item_b) <= 0:
        return False
    return not (
        item_a["z"] + item_a.get("box_z", item_a["height"]) <= item_b["z"]
        or item_b["z"] + item_b.get("box_z", item_b["height"]) <= item_a["z"]
    )


def support_ratio_at_z(item, placed_items, z, tolerance=0.001):
    if z <= tolerance:
        return 1.0

    support_area = 0
    for placed_item in placed_items:
        top_z = placed_item["z"] + placed_item.get("box_z", placed_item["height"])
        if abs(top_z - z) <= tolerance:
            support_area += overlap_area(item, placed_item)

    base_area = item["box_w"] * item["box_h"]
    if base_area <= 0:
        return 0
    return min(1.0, support_area / base_area)


def get_candidate_z_levels(placed_items):
    levels = {0.0}
    for placed_item in placed_items:
        levels.add(
            round(
                placed_item["z"] + placed_item.get("box_z", placed_item["height"]),
                4,
            )
        )
    return sorted(levels)


def get_candidate_axis_positions(placed_items, axis, size_key, pallet_size, box_size):
    positions = {
        0.0,
        round((pallet_size - box_size) / 2, 4),
        round(pallet_size - box_size, 4),
    }
    for placed_item in placed_items:
        start = placed_item[axis]
        end = placed_item[axis] + placed_item[size_key]
        positions.update(
            {
                round(start, 4),
                round(end, 4),
                round(start - box_size, 4),
                round(end - box_size, 4),
            }
        )
    return sorted(pos for pos in positions if 0 <= pos <= pallet_size - box_size)


def get_candidate_positions(placed_items, pallet_length, pallet_width, box_w, box_h):
    x_positions = get_candidate_axis_positions(
        placed_items, "x", "box_w", pallet_length, box_w
    )
    y_positions = get_candidate_axis_positions(
        placed_items, "y", "box_h", pallet_width, box_h
    )
    return [(x, y) for y in y_positions for x in x_positions]


def direct_stack_score(candidate, placed_items, tolerance=0.001):
    for placed_item in placed_items:
        top_z = placed_item["z"] + placed_height(placed_item)
        if abs(top_z - candidate["z"]) > tolerance:
            continue
        same_footprint = (
            abs(placed_item["x"] - candidate["x"]) <= tolerance
            and abs(placed_item["y"] - candidate["y"]) <= tolerance
            and abs(placed_item["box_w"] - candidate["box_w"]) <= tolerance
            and abs(placed_item["box_h"] - candidate["box_h"]) <= tolerance
        )
        same_height = abs(placed_height(placed_item) - placed_height(candidate)) <= tolerance
        if same_footprint and same_height:
            return 1
    return 0


def corner_edge_score(candidate, pallet_length, pallet_width):
    left_gap = candidate["x"]
    right_gap = pallet_length - (candidate["x"] + candidate["box_w"])
    front_gap = candidate["y"]
    back_gap = pallet_width - (candidate["y"] + candidate["box_h"])
    nearest_x_edge = min(left_gap, right_gap)
    nearest_y_edge = min(front_gap, back_gap)
    nearest_corner = min(
        math.hypot(left_gap, front_gap),
        math.hypot(left_gap, back_gap),
        math.hypot(right_gap, front_gap),
        math.hypot(right_gap, back_gap),
    )
    edge_contact_count = sum(
        gap <= 0.001 for gap in (left_gap, right_gap, front_gap, back_gap)
    )
    return (-edge_contact_count, nearest_corner, nearest_x_edge + nearest_y_edge)


def candidate_score(candidate, placed_items, pallet_length, pallet_width):
    top_z = candidate["z"] + candidate.get("box_z", candidate["height"])
    support_ratio = candidate.get("support_ratio", 1)
    base_area = candidate["box_w"] * candidate["box_h"]
    center_x = candidate["x"] + candidate["box_w"] / 2
    center_y = candidate["y"] + candidate["box_h"] / 2
    center_distance = math.hypot(
        center_x - pallet_length / 2,
        center_y - pallet_width / 2,
    )
    edge_score = corner_edge_score(candidate, pallet_length, pallet_width)
    footprint_max_x = max(
        [candidate["x"] + candidate["box_w"]]
        + [item["x"] + item["box_w"] for item in placed_items]
    )
    footprint_max_y = max(
        [candidate["y"] + candidate["box_h"]]
        + [item["y"] + item["box_h"] for item in placed_items]
    )
    if is_gap_fill_item(candidate):
        return (
            candidate["z"],
            base_area,
            top_z,
            footprint_max_x * footprint_max_y,
            edge_score,
            center_distance,
            -support_ratio,
            candidate["y"],
            candidate["x"],
        )

    if candidate["z"] <= 0.001:
        return (
            candidate["z"],
            top_z,
            -base_area,
            edge_score,
            footprint_max_x * footprint_max_y,
            candidate["y"],
            candidate["x"],
            center_distance,
        )

    return (
        top_z,
        candidate["z"],
        -direct_stack_score(candidate, placed_items),
        edge_score,
        footprint_max_x * footprint_max_y,
        center_distance,
        -support_ratio,
        candidate["y"],
        candidate["x"],
    )


def flatten_layer_dict(layer_dict):
    return [
        item
        for items in layer_dict.values()
        for item in items
    ]


def placed_height(item):
    return item.get("box_z", item["height"])


def find_box_position(
    item,
    placed_items,
    pallet_length,
    pallet_width,
    max_stack_height,
    min_support_ratio,
):
    best_candidate = None
    for box_w, box_h, box_z in get_orientations(item, pallet_length, pallet_width):
        for z in get_candidate_z_levels(placed_items):
            if z + box_z > max_stack_height:
                continue

            for x, y in get_candidate_positions(
                placed_items, pallet_length, pallet_width, box_w, box_h
            ):
                candidate = item.copy()
                candidate.update(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "box_w": box_w,
                        "box_h": box_h,
                        "box_z": box_z,
                        "orientation": get_orientation_label(item, box_w, box_h, box_z),
                    }
                )

                if any(
                    boxes_overlap_3d(candidate, placed_item)
                    for placed_item in placed_items
                ):
                    continue

                support_ratio = support_ratio_at_z(candidate, placed_items, z)
                if support_ratio < min_support_ratio:
                    continue

                candidate["support_ratio"] = support_ratio
                candidate["is_stable"] = True
                if best_candidate is None or candidate_score(
                    candidate, placed_items, pallet_length, pallet_width
                ) < candidate_score(
                    best_candidate, placed_items, pallet_length, pallet_width
                ):
                    best_candidate = candidate

    return best_candidate


def forced_candidate_score(candidate, placed_items, pallet_length, pallet_width):
    support_ratio = candidate.get("support_ratio", 0)
    top_z = candidate["z"] + candidate.get("box_z", candidate["height"])
    center_x = candidate["x"] + candidate["box_w"] / 2
    center_y = candidate["y"] + candidate["box_h"] / 2
    center_distance = math.hypot(
        center_x - pallet_length / 2,
        center_y - pallet_width / 2,
    )
    return (
        top_z,
        candidate["z"],
        -support_ratio,
        center_distance,
        candidate["y"],
        candidate["x"],
    )


def force_box_position(item, placed_items, pallet_length, pallet_width, min_support_ratio):
    best_candidate = None
    for box_w, box_h, box_z in get_orientations(item, pallet_length, pallet_width):
        for z in get_candidate_z_levels(placed_items):
            for x, y in get_candidate_positions(
                placed_items, pallet_length, pallet_width, box_w, box_h
            ):
                candidate = item.copy()
                candidate.update(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "box_w": box_w,
                        "box_h": box_h,
                        "box_z": box_z,
                        "orientation": get_orientation_label(item, box_w, box_h, box_z),
                    }
                )
                if any(
                    boxes_overlap_3d(candidate, placed_item)
                    for placed_item in placed_items
                ):
                    continue

                support_ratio = support_ratio_at_z(candidate, placed_items, z)
                candidate["support_ratio"] = support_ratio
                candidate["is_stable"] = support_ratio >= min_support_ratio
                candidate["forced_placement"] = True
                if best_candidate is None or forced_candidate_score(
                    candidate, placed_items, pallet_length, pallet_width
                ) < forced_candidate_score(
                    best_candidate, placed_items, pallet_length, pallet_width
                ):
                    best_candidate = candidate

    if best_candidate is not None:
        return best_candidate

    orientations = get_orientations(item, pallet_length, pallet_width)
    if not orientations:
        return None

    box_w, box_h, box_z = orientations[0]
    highest_z = max(
        (
            placed_item["z"] + placed_item.get("box_z", placed_item["height"])
            for placed_item in placed_items
        ),
        default=0,
    )
    fallback = item.copy()
    fallback.update(
        {
            "x": round((pallet_length - box_w) / 2, 4),
            "y": round((pallet_width - box_h) / 2, 4),
            "z": highest_z,
            "box_w": box_w,
            "box_h": box_h,
            "box_z": box_z,
            "orientation": get_orientation_label(item, box_w, box_h, box_z),
        }
    )
    fallback["support_ratio"] = support_ratio_at_z(fallback, placed_items, highest_z)
    fallback["is_stable"] = fallback["support_ratio"] >= min_support_ratio
    fallback["forced_placement"] = True
    return fallback


def build_height_groups(placed_items, height_tolerance):
    grouped = {}
    group_levels = []
    tolerance = max(0.001, height_tolerance)

    for item in sorted(placed_items, key=lambda box: (box["z"], box["y"], box["x"])):
        group_idx = None
        for idx, level in enumerate(group_levels):
            if abs(item["z"] - level) <= tolerance:
                group_idx = idx
                break

        if group_idx is None:
            group_idx = len(group_levels)
            group_levels.append(item["z"])
            grouped[group_idx] = []

        item["layer_idx"] = group_idx
        grouped[group_idx].append(item)

    return grouped


def pack_single_pallet_3d(
    cargo_items,
    pallet_length,
    pallet_width,
    max_stack_height,
    max_total_weight,
    min_support_ratio,
    height_tolerance,
):
    placed_items = []
    current_weight = 0
    pending_items = sort_for_humanized_placement(cargo_items, mode="foundation")
    round_idx = 0

    while pending_items:
        next_pending_items = []
        placed_this_round = 0

        for item in pending_items:
            if current_weight + item["weight"] > max_total_weight:
                next_pending_items.append(item)
                continue

            placed_item = find_box_position(
                item,
                placed_items,
                pallet_length,
                pallet_width,
                max_stack_height,
                min_support_ratio,
            )
            if placed_item is None:
                next_pending_items.append(item)
                continue

            placed_items.append(placed_item)
            current_weight += item["weight"]
            placed_this_round += 1

        if placed_this_round == 0:
            pending_items = next_pending_items
            break

        round_idx += 1
        next_mode = "foundation" if round_idx == 1 else "gap"
        pending_items = sort_for_humanized_placement(next_pending_items, mode=next_mode)

    return build_height_groups(placed_items, height_tolerance), pending_items


def force_remaining_items_to_pallets(
    pallet_layouts,
    remaining_items,
    pallet_length,
    pallet_width,
    pallet_count,
    min_support_ratio,
    height_tolerance,
):
    still_unplaced = []

    if remaining_items and not pallet_layouts and int(pallet_count) > 0:
        pallet_layouts.append(
            {
                "pallet_idx": 0,
                "layer_dict": {},
                "unstable_items": [],
            }
        )

    for item in sort_for_humanized_placement(remaining_items, mode="gap"):
        target_layout = None
        target_item = None

        for layout in pallet_layouts:
            placed_items = flatten_layer_dict(layout["layer_dict"])
            forced_item = force_box_position(
                item,
                placed_items,
                pallet_length,
                pallet_width,
                min_support_ratio,
            )
            if forced_item is None:
                continue

            if target_item is None or forced_candidate_score(
                forced_item, placed_items, pallet_length, pallet_width
            ) < forced_candidate_score(
                target_item,
                flatten_layer_dict(target_layout["layer_dict"]),
                pallet_length,
                pallet_width,
            ):
                target_layout = layout
                target_item = forced_item

        if target_item is None:
            still_unplaced.append(item)
            continue

        placed_items = flatten_layer_dict(target_layout["layer_dict"])
        placed_items.append(target_item)
        target_layout["layer_dict"] = build_height_groups(placed_items, height_tolerance)

    for layout in pallet_layouts:
        layout["unstable_items"] = [
            item
            for item in flatten_layer_dict(layout["layer_dict"])
            if not item.get("is_stable", True)
        ]

    return still_unplaced


def pack_pallets_3d(
    cargo_items,
    pallet_length,
    pallet_width,
    pallet_count,
    max_stack_height,
    max_total_weight,
    min_support_ratio,
    height_tolerance,
):
    pallet_layouts = []
    remaining_items = cargo_items[:]

    for pallet_idx in range(int(pallet_count)):
        if not remaining_items:
            break

        layer_dict, remaining_after_pallet = pack_single_pallet_3d(
            remaining_items,
            pallet_length,
            pallet_width,
            max_stack_height,
            max_total_weight,
            min_support_ratio,
            height_tolerance,
        )
        if not layer_dict:
            continue

        pallet_layouts.append(
            {
                "pallet_idx": pallet_idx,
                "layer_dict": layer_dict,
                "unstable_items": [],
            }
        )
        remaining_items = remaining_after_pallet

    remaining_items = force_remaining_items_to_pallets(
        pallet_layouts,
        remaining_items,
        pallet_length,
        pallet_width,
        pallet_count,
        min_support_ratio,
        height_tolerance,
    )

    return pallet_layouts, remaining_items


def calculate_pallet_height(layer_dict):
    return max(
        (
            item.get("z", 0) + placed_height(item)
            for items in layer_dict.values()
            for item in items
        ),
        default=0,
    )


def calculate_weight_center(items, pallet_length, pallet_width):
    total_weight = sum(item["weight"] for item in items)
    if total_weight <= 0:
        return 0, 0, 0

    center_x = (
        sum((item["x"] + item["box_w"] / 2) * item["weight"] for item in items)
        / total_weight
    )
    center_y = (
        sum((item["y"] + item["box_h"] / 2) * item["weight"] for item in items)
        / total_weight
    )
    offset_x = center_x - pallet_length / 2
    offset_y = center_y - pallet_width / 2
    offset_distance = math.hypot(offset_x, offset_y)
    return offset_x, offset_y, offset_distance


def build_layer_summary(
    layer_dict,
    pallet_area,
    pallet_length,
    pallet_width,
    pallet_idx,
    min_support_ratio,
):
    summary_rows = []
    for display_idx, layer_idx in enumerate(sorted(layer_dict.keys()), start=1):
        items = layer_dict[layer_idx]
        layer_area = sum(item["area"] for item in items)
        heights = [placed_height(item) for item in items]
        support_ratios = [item.get("support_ratio", 1) for item in items]
        forced_count = sum(item.get("forced_placement", False) for item in items)
        center_offset_x, center_offset_y, center_offset_distance = calculate_weight_center(
            items, pallet_length, pallet_width
        )
        flat_rotated_count = sum(
            item.get("orientation") == "平面轉向90度" for item in items
        )
        vertical_rotated_count = sum(
            item.get("orientation") == "立放/側放" for item in items
        )
        summary_rows.append(
            {
                "棧板": pallet_idx + 1,
                "高度群組": display_idx,
                "箱數": sum(item.get("box_count", 1) for item in items),
                "重量(kg)": round(sum(item["weight"] for item in items), 2),
                "群組最大箱高(cm)": round(max(heights), 2),
                "實際最高Z(cm)": round(
                    max(item.get("z", 0) + placed_height(item) for item in items), 2
                ),
                "高度落差(cm)": round(max(heights) - min(heights), 2),
                "支撐不足數": sum(
                    ratio < min_support_ratio for ratio in support_ratios
                ),
                "強制排入數": forced_count,
                "平面轉向數": flat_rotated_count,
                "立放/側放數": vertical_rotated_count,
                "最低支撐率": f"{min(support_ratios):.0%}",
                "重心X偏移(cm)": round(center_offset_x, 2),
                "重心Y偏移(cm)": round(center_offset_y, 2),
                "重心偏移距離(cm)": round(center_offset_distance, 2),
                "面積利用率": f"{layer_area / pallet_area:.1%}",
            }
        )
    return pd.DataFrame(summary_rows)


def get_layer_footprint(items):
    if not items:
        return 0
    min_x = min(item["x"] for item in items)
    min_y = min(item["y"] for item in items)
    max_x = max(item["x"] + item["box_w"] for item in items)
    max_y = max(item["y"] + item["box_h"] for item in items)
    return max(0, max_x - min_x) * max(0, max_y - min_y)


def build_humanized_stacking_warnings(layer_dict, pallet_area, height_tolerance):
    warnings = []
    sorted_layers = sorted(layer_dict.keys())

    for lower_layer, upper_layer in zip(sorted_layers, sorted_layers[1:]):
        lower_items = layer_dict[lower_layer]
        upper_items = layer_dict[upper_layer]
        lower_weight = sum(item["weight"] for item in lower_items)
        upper_weight = sum(item["weight"] for item in upper_items)

        if upper_weight > lower_weight * 1.1:
            warnings.append(
                f"第 {upper_layer + 1} 高度群組重量高於下方群組較多，可能不符合重物放底層原則。"
            )

        upper_footprint = get_layer_footprint(upper_items)
        if upper_footprint < pallet_area * 0.45 and upper_layer >= 2:
            warnings.append(
                f"第 {upper_layer + 1} 高度群組占地面積偏小，可能形成窄高塔，運輸時較不穩。"
            )

    if sorted_layers:
        top_items = layer_dict[sorted_layers[-1]]
        top_heights = [placed_height(item) for item in top_items]
        if max(top_heights) - min(top_heights) > height_tolerance:
            warnings.append("最高高度群組落差較大，若需要雙層棧板或上方承壓，建議調整為較平整。")

    return warnings


def get_layer_base_heights(layer_dict):
    base_heights = {}
    current_height = 0
    for layer_idx in sorted(layer_dict.keys()):
        base_heights[layer_idx] = current_height
        current_height += max(placed_height(item) for item in layer_dict[layer_idx])
    return base_heights


def cuboid_vertices(x, y, z, length, width, height):
    x1 = x + length
    y1 = y + width
    z1 = z + height
    return [
        (x, y, z),
        (x1, y, z),
        (x1, y1, z),
        (x, y1, z),
        (x, y, z1),
        (x1, y, z1),
        (x1, y1, z1),
        (x, y1, z1),
    ]


def add_box_trace(fig, item, layer_idx, z):
    box_color = item["color"]
    if not item.get("is_stable", True):
        box_color = "#FF4B4B"
    elif item.get("forced_placement", False):
        box_color = "#FFA500"

    vertices = cuboid_vertices(
        item["x"],
        item["y"],
        z,
        item["box_w"],
        item["box_h"],
        placed_height(item),
    )
    xs, ys, zs = zip(*vertices)
    fig.add_trace(
        go.Mesh3d(
            x=xs,
            y=ys,
            z=zs,
            i=[0, 0, 0, 1, 4, 4, 4, 5, 0, 3, 2, 1],
            j=[1, 2, 3, 2, 5, 6, 7, 6, 4, 7, 6, 5],
            k=[2, 3, 0, 6, 6, 7, 4, 1, 5, 2, 1, 0],
            color=box_color,
            opacity=0.72,
            flatshading=True,
            name=item["name"],
            hovertemplate=(
                f"{item['name']}<br>"
                f"Height group: {layer_idx + 1}<br>"
                f"Z: {z:.1f} cm<br>"
                f"Size: {item['box_w']:.1f} x {item['box_h']:.1f} x {placed_height(item):.1f} cm<br>"
                f"Orientation: {item.get('orientation', '原方向')}<br>"
                f"Weight: {item['weight']:.1f} kg<br>"
                f"Support: {item.get('support_ratio', 1):.0%}<br>"
                f"Forced: {'Yes' if item.get('forced_placement', False) else 'No'}<br>"
                f"Planar rotation: {'Yes' if item.get('can_rotate', True) else 'No'}<br>"
                f"Can tip: {'Yes' if item.get('can_tip', False) else 'No'}"
                "<extra></extra>"
            ),
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[item["x"] + item["box_w"] / 2],
            y=[item["y"] + item["box_h"] / 2],
            z=[z + placed_height(item) / 2],
            mode="text",
            text=[item["name"]],
            textfont={"size": 10, "color": "black"},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def draw_3d_stack(layer_dict, pallet_length, pallet_width, title="3D Stack View (可拖曳旋轉)"):
    base_heights = get_layer_base_heights(layer_dict)
    total_height = calculate_pallet_height(layer_dict)

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=[0, pallet_length, pallet_length, 0],
            y=[0, 0, pallet_width, pallet_width],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#F5F5F5",
            opacity=0.35,
            name="棧板",
            showscale=False,
        )
    )

    for layer_idx in sorted(layer_dict.keys()):
        for item in layer_dict[layer_idx]:
            z = item.get("z", base_heights[layer_idx])
            add_box_trace(fig, item, layer_idx, z)

    fig.update_layout(
        title=title,
        height=760,
        scene={
            "xaxis_title": "Length (cm)",
            "yaxis_title": "Width (cm)",
            "zaxis_title": "Height (cm)",
            "xaxis": {"range": [0, pallet_length]},
            "yaxis": {"range": [0, pallet_width]},
            "zaxis": {"range": [0, max(total_height, 1)]},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=False,
    )
    return fig


PRODUCT_PACK_COLUMN = "加入裝箱"
PRODUCT_REQUIRED_COLUMNS = [
    "產品名稱",
    "長度(cm)",
    "寬度(cm)",
    "高度(cm)",
    "數量(件)",
    "單件重量(kg)",
    "顏色代碼(HEX)",
]
AUTO_COLOR_PALETTE = [
    "#8DA0CB",
    "#FC8D62",
    "#66C2A5",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
]


def get_next_auto_color(data_df):
    used_colors = {
        str(value).strip().upper()
        for value in data_df.get("顏色代碼(HEX)", pd.Series(dtype=str)).dropna()
        if str(value).strip()
    }
    for color in AUTO_COLOR_PALETTE:
        if color.upper() not in used_colors:
            return color
    return AUTO_COLOR_PALETTE[len(data_df) % len(AUTO_COLOR_PALETTE)]


def fill_missing_colors(data_df):
    """Assign stable palette colors to rows with a missing HEX color value."""
    completed_df = data_df.copy()
    if "顏色代碼(HEX)" not in completed_df.columns:
        completed_df["顏色代碼(HEX)"] = ""
    for position, row_idx in enumerate(completed_df.index):
        color = completed_df.at[row_idx, "顏色代碼(HEX)"]
        if pd.isna(color) or not str(color).strip() or str(color).strip().lower() == "nan":
            completed_df.at[row_idx, "顏色代碼(HEX)"] = AUTO_COLOR_PALETTE[
                position % len(AUTO_COLOR_PALETTE)
            ]
    return completed_df


def normalize_products(
    product_df,
    carton_length,
    carton_width,
    default_can_rotate=True,
    default_can_tip=False,
):
    """Convert product input columns to the common 3D packing item format."""
    cargo_df = product_df.rename(
        columns={
            PRODUCT_PACK_COLUMN: PACK_COLUMN,
            "產品名稱": "箱型名稱",
            "數量(件)": "數量(箱)",
            "單件重量(kg)": "單箱重量(kg)",
        }
    )
    return normalize_cargo(
        cargo_df,
        carton_length,
        carton_width,
        default_can_rotate,
        default_can_tip,
    )


def pack_cartons_3d(
    product_items,
    carton_length,
    carton_width,
    carton_height,
    max_product_weight,
    max_carton_count,
    min_support_ratio,
):
    """Pack products into closed cartons without any forced/invalid placement."""
    carton_layouts = []
    remaining_items = product_items[:]

    for carton_idx in range(int(max_carton_count)):
        if not remaining_items:
            break
        layer_dict, next_remaining = pack_single_pallet_3d(
            remaining_items,
            carton_length,
            carton_width,
            carton_height,
            max_product_weight,
            min_support_ratio,
            height_tolerance=0.001,
        )
        if not layer_dict:
            break
        carton_layouts.append(
            {
                "carton_idx": carton_idx,
                "layer_dict": layer_dict,
            }
        )
        remaining_items = next_remaining

    return carton_layouts, remaining_items


def draw_3d_carton(layer_dict, carton_length, carton_width, carton_height, title):
    fig = draw_3d_stack(
        layer_dict,
        carton_length,
        carton_width,
        title=title,
    )
    vertices = cuboid_vertices(0, 0, 0, carton_length, carton_width, carton_height)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start_idx, end_idx in edges:
        start = vertices[start_idx]
        end = vertices[end_idx]
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line={"color": "#444444", "width": 4},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        scene={
            "xaxis_title": "Length (cm)",
            "yaxis_title": "Width (cm)",
            "zaxis_title": "Height (cm)",
            "xaxis": {"range": [0, carton_length]},
            "yaxis": {"range": [0, carton_width]},
            "zaxis": {"range": [0, carton_height]},
            "aspectmode": "data",
        }
    )
    return fig


def render_carton_packing_mode():
    st.markdown(
        "輸入產品尺寸、數量與外箱 **內部尺寸**，系統會計算擺放方向、可裝數量、重量與空間利用率。"
    )

    st.sidebar.header("📦 大外箱規格設定")
    carton_length = st.sidebar.number_input("外箱內部長度 (cm)", min_value=0.1, value=60.0)
    carton_width = st.sidebar.number_input("外箱內部寬度 (cm)", min_value=0.1, value=40.0)
    carton_height = st.sidebar.number_input("外箱內部高度 (cm)", min_value=0.1, value=40.0)
    carton_tare_weight = st.sidebar.number_input("外箱空箱重量 (kg)", min_value=0.0, value=1.0)
    max_gross_weight = st.sidebar.number_input("單箱最大毛重 (kg)", min_value=0.1, value=25.0)
    max_carton_count = st.sidebar.number_input(
        "最多可使用外箱數量", min_value=1, max_value=100, value=10
    )
    default_can_rotate = st.sidebar.checkbox("預設允許平面旋轉 90°", value=True)
    default_can_tip = st.sidebar.checkbox(
        "預設允許產品翻面",
        value=False,
        help="易碎品、有固定朝向或可能漏液的產品請勿勾選。",
    )
    min_support_ratio = st.sidebar.slider(
        "最小支撐比例",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )

    default_products = {
        PRODUCT_PACK_COLUMN: [True, True],
        "產品名稱": ["產品 A", "產品 B"],
        "長度(cm)": [20.0, 15.0],
        "寬度(cm)": [10.0, 10.0],
        "高度(cm)": [8.0, 12.0],
        "數量(件)": [12, 6],
        "單件重量(kg)": [0.8, 0.5],
        "顏色代碼(HEX)": ["#8DA0CB", "#FC8D62"],
        ROTATION_COLUMN: [True, True],
        TIPPING_COLUMN: [False, False],
    }
    if "product_packing_df" not in st.session_state:
        st.session_state.product_packing_df = pd.DataFrame(default_products)
    if "product_editor_version" not in st.session_state:
        st.session_state.product_editor_version = 0

    st.subheader("✍️ 產品資料")
    st.caption("精確排法需要產品的長、寬、高；只有總體積或材積重量無法判斷實際是否能放入。")
    product_df = st.data_editor(
        st.session_state.product_packing_df,
        num_rows="dynamic",
        column_config={
            PRODUCT_PACK_COLUMN: st.column_config.CheckboxColumn("加入裝箱", default=True),
            ROTATION_COLUMN: st.column_config.CheckboxColumn("可平面旋轉", default=True),
            TIPPING_COLUMN: st.column_config.CheckboxColumn("可翻面", default=False),
            "顏色代碼(HEX)": st.column_config.TextColumn(
                "顏色代碼(HEX)",
                help="可輸入例如 #8DA0CB；也可使用下方快速新增產品的選色器。",
            ),
        },
        key=f"product_packing_editor_{st.session_state.product_editor_version}",
        use_container_width=True,
    )

    with st.expander("➕ 快速新增產品（可直接選顏色）"):
        with st.form("quick_add_product_form", clear_on_submit=True):
            input_cols = st.columns(4)
            new_product_name = input_cols[0].text_input("產品名稱")
            new_product_length = input_cols[1].number_input(
                "長度 (cm)", min_value=0.1, value=10.0
            )
            new_product_width = input_cols[2].number_input(
                "寬度 (cm)", min_value=0.1, value=10.0
            )
            new_product_height = input_cols[3].number_input(
                "高度 (cm)", min_value=0.1, value=10.0
            )
            detail_cols = st.columns(5)
            new_product_count = detail_cols[0].number_input(
                "數量 (件)", min_value=1, value=1, step=1
            )
            new_product_weight = detail_cols[1].number_input(
                "單件重量 (kg)", min_value=0.01, value=1.0
            )
            new_product_color = detail_cols[2].color_picker(
                "產品顏色", value=get_next_auto_color(product_df)
            )
            new_can_rotate = detail_cols[3].checkbox("可平面旋轉", value=True)
            new_can_tip = detail_cols[4].checkbox("可翻面", value=False)
            add_product = st.form_submit_button("新增到產品表格", type="primary")

        if add_product:
            if not new_product_name.strip():
                st.error("請輸入產品名稱。")
            else:
                new_row = pd.DataFrame(
                    [
                        {
                            PRODUCT_PACK_COLUMN: True,
                            "產品名稱": new_product_name.strip(),
                            "長度(cm)": float(new_product_length),
                            "寬度(cm)": float(new_product_width),
                            "高度(cm)": float(new_product_height),
                            "數量(件)": int(new_product_count),
                            "單件重量(kg)": float(new_product_weight),
                            "顏色代碼(HEX)": new_product_color,
                            ROTATION_COLUMN: new_can_rotate,
                            TIPPING_COLUMN: new_can_tip,
                        }
                    ]
                )
                st.session_state.product_packing_df = pd.concat(
                    [product_df, new_row], ignore_index=True
                )
                st.session_state.product_editor_version += 1
                st.rerun()

    if not st.button("🧮 開始產品裝箱計算", type="primary"):
        return

    product_df = fill_missing_colors(product_df)

    missing_columns = [col for col in PRODUCT_REQUIRED_COLUMNS if col not in product_df.columns]
    if missing_columns:
        st.error(f"缺少必要欄位：{', '.join(missing_columns)}")
        return
    if PRODUCT_PACK_COLUMN not in product_df.columns or not product_df[
        PRODUCT_PACK_COLUMN
    ].apply(lambda value: parse_bool(value, default=True)).any():
        st.error("請至少勾選一項要加入裝箱的產品。")
        return

    max_product_weight = max_gross_weight - carton_tare_weight
    if max_product_weight <= 0:
        st.error("單箱最大毛重必須大於外箱空箱重量。")
        return

    product_items, invalid_rows, base_oversized_items = normalize_products(
        product_df,
        carton_length,
        carton_width,
        default_can_rotate,
        default_can_tip,
    )
    if invalid_rows:
        st.warning(f"以下資料列格式不正確，已略過：{invalid_rows}")

    with st.spinner("正在尋找 3D 裝箱擺法..."):
        carton_layouts, unpacked_items = pack_cartons_3d(
            product_items,
            carton_length,
            carton_width,
            carton_height,
            max_product_weight,
            int(max_carton_count),
            min_support_ratio,
        )

    unpacked_items = base_oversized_items + unpacked_items
    packed_items = [
        item
        for layout in carton_layouts
        for item in flatten_layer_dict(layout["layer_dict"])
    ]
    requested_count = len(product_items) + len(base_oversized_items)
    packed_count = len(packed_items)
    unpacked_count = len(unpacked_items)
    total_net_weight = sum(item["weight"] for item in packed_items)
    total_gross_weight = total_net_weight + len(carton_layouts) * carton_tare_weight
    carton_volume = carton_length * carton_width * carton_height
    packed_volume = sum(item["volume"] for item in packed_items)
    overall_utilization = (
        packed_volume / (carton_volume * len(carton_layouts))
        if carton_layouts
        else 0
    )

    if requested_count and unpacked_count == 0:
        st.success(f"全部 {requested_count} 件產品可裝入，共需要 {len(carton_layouts)} 個外箱。")
    elif packed_count:
        st.warning(
            f"目前可裝入 {packed_count} / {requested_count} 件，仍有 {unpacked_count} 件無法裝入。"
        )
    else:
        st.error("沒有產品能在目前的尺寸、方向與重量限制下裝入外箱。")

    metric_cols = st.columns(5)
    metric_cols[0].metric("已裝件數", f"{packed_count} / {requested_count}")
    metric_cols[1].metric("使用外箱", len(carton_layouts))
    metric_cols[2].metric("產品淨重", f"{total_net_weight:.2f} kg")
    metric_cols[3].metric("合計毛重", f"{total_gross_weight:.2f} kg")
    metric_cols[4].metric("平均空間利用率", f"{overall_utilization:.1%}")

    if unpacked_items:
        unpacked_summary = {}
        for item in unpacked_items:
            unpacked_summary[item["name"]] = unpacked_summary.get(item["name"], 0) + 1
        st.warning(
            "未裝入產品："
            + "、".join(f"{name} {count} 件" for name, count in unpacked_summary.items())
            + "。可能原因為尺寸、可用箱數、承重或擺放方向限制。"
        )

    if not carton_layouts:
        return

    summary_rows = []
    for layout in carton_layouts:
        items = flatten_layer_dict(layout["layer_dict"])
        net_weight = sum(item["weight"] for item in items)
        summary_rows.append(
            {
                "外箱": layout["carton_idx"] + 1,
                "產品件數": len(items),
                "產品淨重(kg)": round(net_weight, 2),
                "外箱毛重(kg)": round(net_weight + carton_tare_weight, 2),
                "實際使用高度(cm)": round(calculate_pallet_height(layout["layer_dict"]), 2),
                "空間利用率": f"{sum(item['volume'] for item in items) / carton_volume:.1%}",
            }
        )
    st.subheader("📊 外箱摘要")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.subheader("🧊 3D 裝箱擺法")
    for layout in carton_layouts:
        carton_no = layout["carton_idx"] + 1
        items = flatten_layer_dict(layout["layer_dict"])
        with st.expander(f"外箱 {carton_no}｜{len(items)} 件產品", expanded=carton_no == 1):
            st.plotly_chart(
                draw_3d_carton(
                    layout["layer_dict"],
                    carton_length,
                    carton_width,
                    carton_height,
                    title=f"外箱 {carton_no} - 3D 裝箱配置",
                ),
                use_container_width=True,
            )
            position_rows = [
                {
                    "產品": item["name"],
                    "X(cm)": item["x"],
                    "Y(cm)": item["y"],
                    "Z(cm)": item["z"],
                    "擺放尺寸(cm)": f"{item['box_w']:g} × {item['box_h']:g} × {placed_height(item):g}",
                    "方向": item.get("orientation", "原方向"),
                    "重量(kg)": item["weight"],
                }
                for item in sorted(items, key=lambda value: (value["z"], value["y"], value["x"]))
            ]
            st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)


st.set_page_config(page_title="智慧 3D 物流裝載優化系統", layout="wide")

st.title("📦 智慧 3D 物流裝載優化系統")
operation_mode = st.radio(
    "功能模式",
    ["🧱 棧板打板", "📦 產品裝箱"],
    horizontal=True,
)

if operation_mode == "📦 產品裝箱":
    render_carton_packing_mode()
    st.stop()

st.markdown(
    "本系統改用 **逐箱 3D 打板邏輯**：每一箱會依棧板面與已放置箱子的上表面尋找位置，不再先假設完整平面層。"
)

st.sidebar.header("📐 棧板規格設定")
pallet_length = st.sidebar.number_input("棧板長度 (cm)", min_value=1.0, value=124.0)
pallet_width = st.sidebar.number_input("棧板寬度 (cm)", min_value=1.0, value=95.0)
pallet_count = st.sidebar.number_input("棧板數量", min_value=1, max_value=50, value=1)
max_stack_height = st.sidebar.number_input("允許總高度 (cm)", min_value=1.0, value=160.0)
max_total_weight = st.sidebar.number_input("允許單棧板重量 (kg)", min_value=1.0, value=1000.0)
max_layer_weight = st.sidebar.number_input("高度群組重量警戒值 (kg)", min_value=1.0, value=300.0)
default_can_rotate = st.sidebar.checkbox(
    "預設允許平面旋轉 90°",
    value=True,
    help="箱高保持不變，只交換箱子的長與寬；可利用棧板右側等狹長空間。",
)
default_can_tip = st.sidebar.checkbox(
    "預設允許箱子翻面",
    value=False,
    help="勾選後才允許改用箱子的長或寬作為高度。一般紙箱建議維持不勾選。",
)
min_support_ratio = st.sidebar.slider(
    "最小支撐比例",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
)
height_tolerance = st.sidebar.number_input("高度群組容許落差 (cm)", min_value=0.0, value=3.0)
center_tolerance = st.sidebar.number_input("重心偏移警戒值 (cm)", min_value=0.0, value=10.0)

input_method = st.radio("請選擇數據輸入方式：", ["📋 手動在網頁輸入數據", "📁 上傳 Excel/CSV Packing List"])

cargo_df = pd.DataFrame()
editor_column_config = {
    PACK_COLUMN: st.column_config.CheckboxColumn(
        "排入棧板",
        help="只會計算有勾選的箱型。可只勾選相同尺寸的箱子，先查看它們排在同一棧板的結果。",
        default=True,
    ),
    ROTATION_COLUMN: st.column_config.CheckboxColumn(
        "可平面旋轉",
        help="勾選後可在棧板平面旋轉 90°，箱高維持不變。",
        default=True,
    ),
    TIPPING_COLUMN: st.column_config.CheckboxColumn(
        "可翻面",
        help="勾選後才允許以箱子的其他邊作為高度。",
        default=False,
    ),
    "顏色代碼(HEX)": st.column_config.TextColumn(
        "顏色代碼(HEX)",
        help="可自行輸入例如 #8DA0CB；留白時系統會自動分配顏色。",
    ),
}

if input_method == "📋 手動在網頁輸入數據":
    st.subheader("✍️ 請輸入各箱型數據")
    st.caption("使用「排入棧板」勾選本次要計算的箱型；新增箱型若未填顏色代碼，系統會自動分配顏色。")
    default_data = {
        PACK_COLUMN: [True, True, True, True, True],
        "箱型名稱": ["大箱", "中箱", "長條A", "長條B", "薄箱"],
        "長度(cm)": [54, 37, 88, 67, 37],
        "寬度(cm)": [45, 32, 23, 23, 32],
        "高度(cm)": [37, 30, 17, 23, 14],
        "數量(箱)": [6, 8, 2, 3, 8],
        "單箱重量(kg)": [21.0, 5.7, 16.7, 19.3, 13.0],
        "顏色代碼(HEX)": ["#E5C494", "#8DA0CB", "#FC8D62", "#A6D854", "#FFD92F"],
        ROTATION_COLUMN: [True, True, True, True, True],
        TIPPING_COLUMN: [False, False, False, False, False],
    }
    if "manual_cargo_df" not in st.session_state:
        st.session_state.manual_cargo_df = pd.DataFrame(default_data)
    cargo_df = st.data_editor(
        st.session_state.manual_cargo_df,
        num_rows="dynamic",
        column_config=editor_column_config,
        key="manual_cargo_editor",
    )
else:
    st.subheader("📁 上傳 Packing List 檔案")
    uploaded_file = st.file_uploader("請選擇您的 Excel 或 CSV 檔案", type=["xlsx", "csv"])
    if uploaded_file is not None:
        upload_identity = (
            uploaded_file.name,
            uploaded_file.size,
            getattr(uploaded_file, "file_id", None),
        )
        if (
            st.session_state.get("uploaded_file_identity") != upload_identity
            or "uploaded_cargo_df" not in st.session_state
        ):
            if uploaded_file.name.endswith(".csv"):
                uploaded_cargo_df = pd.read_csv(uploaded_file)
            else:
                uploaded_cargo_df = pd.read_excel(uploaded_file)
            if PACK_COLUMN not in uploaded_cargo_df.columns:
                uploaded_cargo_df.insert(0, PACK_COLUMN, True)
            if ROTATION_COLUMN not in uploaded_cargo_df.columns:
                if LEGACY_ROTATION_COLUMN in uploaded_cargo_df.columns:
                    uploaded_cargo_df[ROTATION_COLUMN] = uploaded_cargo_df[
                        LEGACY_ROTATION_COLUMN
                    ]
                else:
                    uploaded_cargo_df[ROTATION_COLUMN] = default_can_rotate
            if TIPPING_COLUMN not in uploaded_cargo_df.columns:
                uploaded_cargo_df[TIPPING_COLUMN] = default_can_tip
            if LEGACY_ROTATION_COLUMN in uploaded_cargo_df.columns:
                uploaded_cargo_df = uploaded_cargo_df.drop(
                    columns=[LEGACY_ROTATION_COLUMN]
                )
            uploaded_cargo_df = fill_missing_colors(uploaded_cargo_df)
            st.session_state.uploaded_cargo_df = uploaded_cargo_df
            st.session_state.uploaded_file_identity = upload_identity
            st.session_state.pop("uploaded_cargo_editor", None)
        st.write("📊 偵測到的 Packing List 內容：")
        st.caption("使用「排入棧板」勾選本次要計算的箱型；未勾選的資料仍會保留在表格中，但不會參與本次打板。")
        cargo_df = st.data_editor(
            st.session_state.uploaded_cargo_df,
            num_rows="dynamic",
            column_config=editor_column_config,
            key="uploaded_cargo_editor",
        )

if not cargo_df.empty and st.button("🚀 開始智慧自動打板計算"):
    cargo_df = fill_missing_colors(cargo_df)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in cargo_df.columns]
    if missing_columns:
        st.error(f"缺少必要欄位：{', '.join(missing_columns)}")
        st.stop()
    if not cargo_df[PACK_COLUMN].apply(
        lambda value: parse_bool(value, default=True)
    ).any():
        st.error("請至少勾選一個要排入棧板的箱型。")
        st.stop()

    try:
        pallet_area = pallet_length * pallet_width
        cargo_items, invalid_rows, oversized_items = normalize_cargo(
            cargo_df,
            pallet_length,
            pallet_width,
            default_can_rotate,
            default_can_tip,
        )

        with st.spinner("正在計算多棧板逐箱 3D 打板方案，請稍候..."):
            pallet_layouts, unpacked_items = pack_pallets_3d(
                cargo_items,
                pallet_length,
                pallet_width,
                int(pallet_count),
                max_stack_height,
                max_total_weight,
                min_support_ratio,
                height_tolerance,
            )

        total_height_groups = sum(len(layout["layer_dict"]) for layout in pallet_layouts)
        used_pallet_count = len(pallet_layouts)

        if invalid_rows:
            st.warning(f"以下資料列格式錯誤，已略過：{invalid_rows}")
        if oversized_items:
            st.warning(f"有 {len(oversized_items)} 箱超過棧板尺寸，無法排入。")
        if unpacked_items:
            unpacked_box_count = sum(item.get("box_count", 1) for item in unpacked_items)
            st.warning(
                f"仍有 {unpacked_box_count} 箱無法排入，通常代表該箱型長寬無法放進棧板平面。"
            )

        if not pallet_layouts:
            st.error("沒有任何箱子成功排入棧板，請檢查尺寸、重量、高度與支撐設定。")
            st.stop()

        summary_frames = [
            build_layer_summary(
                layout["layer_dict"],
                pallet_area,
                pallet_length,
                pallet_width,
                layout["pallet_idx"],
                min_support_ratio,
            )
            for layout in pallet_layouts
        ]
        summary_df = pd.concat(summary_frames, ignore_index=True)
        pallet_heights = [
            calculate_pallet_height(layout["layer_dict"]) for layout in pallet_layouts
        ]
        pallet_weights = [
            sum(
                item["weight"]
                for items in layout["layer_dict"].values()
                for item in items
            )
            for layout in pallet_layouts
        ]
        unstable_box_count = sum(
            len(layout["unstable_items"]) for layout in pallet_layouts
        )
        forced_box_count = sum(
            item.get("forced_placement", False)
            for layout in pallet_layouts
            for item in flatten_layer_dict(layout["layer_dict"])
        )
        total_boxes = int(summary_df["箱數"].sum())
        total_weight = float(summary_df["重量(kg)"].sum())
        max_pallet_height = max(pallet_heights, default=0)
        avg_utilization = summary_df["面積利用率"].str.rstrip("%").astype(float).mean() / 100

        st.success(
            f"🎯 計算完成！成功排入 {total_boxes} 箱，使用 {used_pallet_count} 個棧板，共 {total_height_groups} 個高度群組。"
        )

        metric_cols = st.columns(5)
        metric_cols[0].metric("使用棧板", f"{used_pallet_count} / {int(pallet_count)}")
        metric_cols[1].metric("高度群組", f"{total_height_groups}")
        metric_cols[2].metric("最高棧板高度", f"{max_pallet_height:.1f} cm")
        metric_cols[3].metric("總重量", f"{total_weight:.1f} kg")
        metric_cols[4].metric("平均面積利用率", f"{avg_utilization:.1%}")

        for idx, (height, weight) in enumerate(zip(pallet_heights, pallet_weights), start=1):
            if height > max_stack_height:
                st.warning(
                    f"第 {idx} 個棧板高度 {height:.1f} cm，超過設定限制 {max_stack_height:.1f} cm。"
                )
            if weight > max_total_weight:
                st.error(
                    f"第 {idx} 個棧板重量 {weight:.1f} kg，超過單棧板限制 {max_total_weight:.1f} kg。"
                )
        if unstable_box_count:
            st.warning(
                f"有 {unstable_box_count} 箱的支撐比例低於 {min_support_ratio:.0%}，3D 圖會以紅色標示。"
            )
        if forced_box_count:
            st.warning(
                f"有 {forced_box_count} 箱為強制排入，可能超過原本高度、重量或支撐限制，3D 圖會以橘色或紅色標示。"
            )

        heavy_layers = summary_df[summary_df["重量(kg)"] > max_layer_weight]
        uneven_layers = summary_df[summary_df["高度落差(cm)"] > height_tolerance]
        off_center_layers = summary_df[
            summary_df["重心偏移距離(cm)"] > center_tolerance
        ]
        humanized_warnings = []
        for layout in pallet_layouts:
            humanized_warnings.extend(
                build_humanized_stacking_warnings(
                    layout["layer_dict"], pallet_area, height_tolerance
                )
            )
        if not heavy_layers.empty:
            st.warning("部分高度群組重量超過警戒值，建議調整排列或降低單箱重量。")
        if not uneven_layers.empty:
            st.warning("部分高度群組內箱高落差較大，上方支撐可能不穩。")
        if not off_center_layers.empty:
            st.warning("部分高度群組重量重心偏離棧板中心較多，建議調整左右或前後配置。")
        for warning_text in humanized_warnings:
            st.warning(warning_text)

        st.subheader("📊 高度群組檢核表")
        st.dataframe(summary_df, use_container_width=True)

        plot_layers = [
            (layout["pallet_idx"], layer_idx, items)
            for layout in pallet_layouts
            for layer_idx, items in sorted(layout["layer_dict"].items())
        ]
        cols = 2
        rows = math.ceil(total_height_groups / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, (plot_pallet_idx, layer_idx, items) in enumerate(plot_layers):
            ax = axes[i]
            pallet = patches.Rectangle(
                (0, 0),
                pallet_length,
                pallet_width,
                linewidth=2,
                edgecolor="#1A1A1A",
                facecolor="#F5F5F5",
                zorder=1,
            )
            ax.add_patch(pallet)

            layer_height = max(placed_height(item) for item in items)
            layer_weight = sum(item["weight"] for item in items)
            center_offset_x, center_offset_y, _ = calculate_weight_center(
                items, pallet_length, pallet_width
            )
            gravity_x = pallet_length / 2 + center_offset_x
            gravity_y = pallet_width / 2 + center_offset_y

            for item in items:
                box = patches.Rectangle(
                    (item["x"], item["y"]),
                    item["box_w"],
                    item["box_h"],
                    linewidth=1.2,
                    edgecolor="#333333",
                    facecolor=item["color"],
                    alpha=0.85,
                    zorder=2,
                )
                ax.add_patch(box)
                ax.text(
                    item["x"] + item["box_w"] / 2,
                    item["y"] + item["box_h"] / 2,
                    f"{item['name']}\n{item['weight']}kg\nH:{placed_height(item)}cm",
                    color="black",
                    weight="bold",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=3,
                )

            ax.scatter(
                [gravity_x],
                [gravity_y],
                color="red",
                marker="x",
                s=100,
                zorder=4,
                label="重心",
            )
            ax.set_xlim(0, pallet_length)
            ax.set_ylim(0, pallet_width)
            ax.set_aspect("equal")
            ax.set_title(
                f"棧板 {plot_pallet_idx + 1} / 高度群組 {layer_idx + 1} | 最高箱高 {layer_height:.1f} cm | 重 {layer_weight:.1f} kg"
            )
            ax.set_xlabel("Length (cm)")
            ax.set_ylabel("Width (cm)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        for j in range(total_height_groups, len(axes)):
            axes[j].axis("off")

        st.subheader("🧱 2D 高度群組俯視圖")
        st.pyplot(fig)

        st.subheader("🧊 可旋轉 3D 逐箱堆疊圖")
        for layout in pallet_layouts:
            pallet_idx = layout["pallet_idx"]
            st.plotly_chart(
                draw_3d_stack(
                    layout["layer_dict"],
                    pallet_length,
                    pallet_width,
                    title=f"棧板 {pallet_idx + 1} - 3D 重力堆疊圖",
                ),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"計算過程發生錯誤：{exc}")
