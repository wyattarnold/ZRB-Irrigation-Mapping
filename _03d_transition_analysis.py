# %% [markdown]
# # Step 3d: Transition Analysis (Year-to-Year Class Changes)
#
# Computes class transition areas between consecutive yearly classified assets,
# exports transition edge/matrix tables, and generates Sankey diagrams.

# %%
import sys
import json
from pathlib import Path

import ee
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib import colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))
import __config__ as config

from src.gee_utils import initialize_earth_engine, asset_exists, _retry_with_backoff


TRANSITION_CONFIG = config.TRANSITION_ANALYSIS
EMB_CONFIG = config.EMBEDDINGS_CLASSIFICATION
CONSOL_CONFIG = config.CONSOLIDATION

OUTPUT_DIR = Path('output') / config.CURRENT_STUDY_AREA / TRANSITION_CONFIG.get('output_subfolder', 'embeddings_classification/transitions')
MATRICES_DIR = OUTPUT_DIR / 'matrices'
MATRICES_MAJOR_DIR = OUTPUT_DIR / 'matrices_major'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATRICES_DIR.mkdir(parents=True, exist_ok=True)
MATRICES_MAJOR_DIR.mkdir(parents=True, exist_ok=True)

print('Initializing Earth Engine...')
initialize_earth_engine()

study_area_config = config.get_study_area()
study_area = study_area_config['geometry']
study_area_name = study_area_config['name']


def _int_key_dict(input_dict):
    return {int(key): value for key, value in (input_dict or {}).items()}


def _hex_to_rgba_tuple(hex_color, alpha=0.45):
    try:
        red, green, blue = mcolors.to_rgb(hex_color or '#999999')
    except ValueError:
        red, green, blue = mcolors.to_rgb('#999999')
    return (red, green, blue, alpha)


def _class_band_name(image):
    band_names = _retry_with_backoff(lambda: image.bandNames().getInfo())
    if 'classification' in band_names:
        return 'classification'
    if not band_names:
        raise ValueError('Image has no bands')
    return band_names[0]


def _schema_settings(schema_name):
    if schema_name == 'aggregated':
        labels = _int_key_dict(TRANSITION_CONFIG.get('aggregated_labels', CONSOL_CONFIG.get('aggregated_labels', {})))
        palette = _int_key_dict(TRANSITION_CONFIG.get('aggregated_palette', CONSOL_CONFIG.get('aggregated_palette', {})))
        class_types = {
            value: value for value in labels.values()
        }
    else:
        labels = _int_key_dict(TRANSITION_CONFIG.get('class_labels', EMB_CONFIG.get('cluster_labels', {})))
        palette = _int_key_dict(TRANSITION_CONFIG.get('class_palette', EMB_CONFIG.get('cluster_palette', {})))
        class_types = TRANSITION_CONFIG.get('class_types', EMB_CONFIG.get('class_types', {}))

    class_ids = sorted(labels.keys())
    if not class_ids:
        raise ValueError('No class labels found for transition schema.')

    class_to_type = {}
    for type_name, class_list in class_types.items():
        for class_id in class_list:
            class_to_type[int(class_id)] = type_name

    return labels, palette, class_to_type, class_ids


def _prepare_class_image(asset_id, schema_name, class_aggregation):
    image = ee.Image(asset_id)
    band_name = _class_band_name(image)
    class_image = image.select(band_name).toInt16()

    if schema_name == 'aggregated':
        source_ids = sorted(class_aggregation.keys())
        target_ids = [class_aggregation[source_id] for source_id in source_ids]
        class_image = class_image.remap(source_ids, target_ids, -1).rename('classification').toInt16()
        class_image = class_image.updateMask(class_image.gte(0))
    else:
        class_image = class_image.rename('classification')

    return class_image


def _compute_transition_groups(image_from, image_to, region, scale, multiplier):
    transition_code = image_from.multiply(multiplier).add(image_to).toInt32().rename('transition')
    pixel_area = ee.Image.pixelArea().divide(1e6).rename('area_km2')
    reducer_input = pixel_area.addBands(transition_code)

    result = _retry_with_backoff(
        lambda: reducer_input.reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName='transition'),
            geometry=region,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        ).getInfo()
    )

    return result.get('groups', [])


def _compute_transition_groups_with_area(transition_code_image, area_image, region, scale):
    reducer_input = area_image.rename('area_km2').addBands(transition_code_image.rename('transition'))
    result = _retry_with_backoff(
        lambda: reducer_input.reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName='transition'),
            geometry=region,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        ).getInfo()
    )
    return result.get('groups', [])


def _probability_band_name(class_label):
    return f"prob_{class_label.lower().replace(' ', '_').replace('.', '')}"


def _select_class_probability(probability_image, class_image, class_ids, band_by_class):
    selected = ee.Image.constant(0).toFloat()
    for class_id in class_ids:
        band_name = band_by_class.get(class_id)
        if band_name is None:
            continue
        selected = selected.add(
            class_image.eq(class_id).multiply(probability_image.select(band_name).toFloat().divide(100.0))
        )
    return selected.rename('selected_prob')


def _draw_pairwise_alluvial(
    axis,
    pair_df,
    year_from,
    year_to,
    class_labels,
    class_palette,
    class_ids,
    left_labels=None,
    right_labels=None,
):
    if pair_df.empty:
        return False

    ids_present = sorted(set(pair_df['source_id']).union(set(pair_df['target_id'])))
    ids_ordered = [class_id for class_id in class_ids if class_id in ids_present]
    if not ids_ordered:
        return False

    source_totals = pair_df.groupby('source_id')['area_km2'].sum().to_dict()
    target_totals = pair_df.groupby('target_id')['area_km2'].sum().to_dict()
    total_area = max(sum(source_totals.values()), sum(target_totals.values()))
    if total_area <= 0:
        return False

    gap = total_area * 0.01

    def _build_positions(totals):
        positions = {}
        cursor = 0.0
        for class_id in ids_ordered:
            height = float(totals.get(class_id, 0.0))
            positions[class_id] = (cursor, cursor + height)
            cursor += height + gap
        return positions, max(cursor - gap, 0.0)

    left_positions, left_height = _build_positions(source_totals)
    right_positions, right_height = _build_positions(target_totals)
    max_height = max(left_height, right_height)

    left_cursor = {class_id: left_positions[class_id][0] for class_id in ids_ordered}
    right_cursor = {class_id: right_positions[class_id][0] for class_id in ids_ordered}

    left_x = 0.08
    right_x = 0.92
    bar_width = 0.05

    ordered_edges = pair_df.sort_values(['source_id', 'target_id', 'area_km2'], ascending=[True, True, False])

    for row in ordered_edges.itertuples(index=False):
        source_id = int(row.source_id)
        target_id = int(row.target_id)
        area_val = float(row.area_km2)
        if area_val <= 0 or source_id not in left_cursor or target_id not in right_cursor:
            continue

        left_bottom = left_cursor[source_id]
        left_top = left_bottom + area_val
        right_bottom = right_cursor[target_id]
        right_top = right_bottom + area_val
        left_cursor[source_id] = left_top
        right_cursor[target_id] = right_top

        polygon_points = [
            (left_x + bar_width, left_bottom),
            (right_x, right_bottom),
            (right_x, right_top),
            (left_x + bar_width, left_top),
        ]
        axis.add_patch(
            Polygon(
                polygon_points,
                closed=True,
                facecolor=_hex_to_rgba_tuple(class_palette.get(source_id, '#999999'), alpha=0.35),
                edgecolor='none',
            )
        )

    for class_id in ids_ordered:
        left_bottom, left_top = left_positions[class_id]
        right_bottom, right_top = right_positions[class_id]
        class_color = class_palette.get(class_id, '#999999')
        class_label_left = left_labels.get(class_id, class_labels.get(class_id, f'Class {class_id}')) if left_labels else class_labels.get(class_id, f'Class {class_id}')
        class_label_right = right_labels.get(class_id, class_labels.get(class_id, f'Class {class_id}')) if right_labels else class_labels.get(class_id, f'Class {class_id}')

        axis.add_patch(Rectangle((left_x, left_bottom), bar_width, left_top - left_bottom, facecolor=class_color, edgecolor='white', linewidth=0.5))
        axis.add_patch(Rectangle((right_x, right_bottom), bar_width, right_top - right_bottom, facecolor=class_color, edgecolor='white', linewidth=0.5))

        axis.text(
            left_x - 0.01,
            (left_bottom + left_top) / 2,
            class_label_left,
            ha='right',
            va='center',
            fontsize=7,
        )
        axis.text(
            right_x + bar_width + 0.01,
            (right_bottom + right_top) / 2,
            class_label_right,
            ha='left',
            va='center',
            fontsize=7,
        )

    axis.text(left_x + bar_width / 2, max_height * 1.03, str(year_from), ha='center', va='bottom', fontsize=9, fontweight='bold')
    axis.text(right_x + bar_width / 2, max_height * 1.03, str(year_to), ha='center', va='bottom', fontsize=9, fontweight='bold')
    axis.set_xlim(0, 1.05)
    axis.set_ylim(0, max_height * 1.08)
    axis.axis('off')
    return True


def _save_pairwise_alluvial(
    pair_df,
    year_from,
    year_to,
    class_labels,
    class_palette,
    class_ids,
    output_path,
    figure_width=10.0,
    include_node_totals=False,
):
    source_totals = pair_df.groupby('source_id')['area_km2'].sum().to_dict() if not pair_df.empty else {}
    target_totals = pair_df.groupby('target_id')['area_km2'].sum().to_dict() if not pair_df.empty else {}

    left_labels = None
    right_labels = None
    if include_node_totals:
        left_labels = {
            class_id: f"{class_labels.get(class_id, f'Class {class_id}')} ({source_totals.get(class_id, 0):,.0f} km²)"
            for class_id in class_ids
        }
        right_labels = {
            class_id: f"{class_labels.get(class_id, f'Class {class_id}')} ({target_totals.get(class_id, 0):,.0f} km²)"
            for class_id in class_ids
        }

    figure_height = max(3.2, 0.42 * len(class_ids))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    created = _draw_pairwise_alluvial(
        axis,
        pair_df,
        year_from,
        year_to,
        class_labels,
        class_palette,
        class_ids,
        left_labels=left_labels,
        right_labels=right_labels,
    )
    if not created:
        plt.close(figure)
        return False
    figure.suptitle(f'Transitions {year_from} → {year_to} (km²)', fontsize=12, fontweight='bold')
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return True


def _save_left_to_right_alluvial(edges_df, years, class_labels, class_palette, class_ids, output_path, title):
    if edges_df.empty:
        return False

    transition_pairs = list(zip(years[:-1], years[1:]))
    if not transition_pairs:
        return False

    ids_present = sorted(set(edges_df['source_id']).union(set(edges_df['target_id'])))
    ids_ordered = [class_id for class_id in class_ids if class_id in ids_present]
    if not ids_ordered:
        return False

    totals_by_year = {}
    for index, year in enumerate(years):
        incoming = edges_df[edges_df['year_to'] == year].groupby('target_id')['area_km2'].sum().to_dict()
        outgoing = edges_df[edges_df['year_from'] == year].groupby('source_id')['area_km2'].sum().to_dict()
        year_totals = {}
        for class_id in ids_ordered:
            in_val = float(incoming.get(class_id, 0.0))
            out_val = float(outgoing.get(class_id, 0.0))
            if index == 0:
                year_totals[class_id] = out_val
            elif index == len(years) - 1:
                year_totals[class_id] = in_val
            else:
                year_totals[class_id] = max(in_val, out_val)
        totals_by_year[year] = year_totals

    max_total = max(sum(year_totals.values()) for year_totals in totals_by_year.values())
    if max_total <= 0:
        return False

    gap = max_total * 0.01
    positions = {}
    max_stack_height = 0.0
    for year in years:
        cursor = 0.0
        year_positions = {}
        for class_id in ids_ordered:
            height = totals_by_year[year].get(class_id, 0.0)
            year_positions[class_id] = (cursor, cursor + height)
            cursor += height + gap
        positions[year] = year_positions
        max_stack_height = max(max_stack_height, max(cursor - gap, 0.0))

    x_start = 0.08
    x_end = 0.92
    x_positions = {
        year: x_start + idx * (x_end - x_start) / (len(years) - 1)
        for idx, year in enumerate(years)
    }
    bar_width = 0.028

    out_cursor = {year: {class_id: positions[year][class_id][0] for class_id in ids_ordered} for year in years}
    in_cursor = {year: {class_id: positions[year][class_id][0] for class_id in ids_ordered} for year in years}

    figure_width = float(TRANSITION_CONFIG.get('multiyear_flow_width_in', 6.5))
    figure_height = max(3.8, min(7.0, 2.2 + 0.28 * len(ids_ordered)))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))

    for year_from, year_to in transition_pairs:
        pair_df = edges_df[(edges_df['year_from'] == year_from) & (edges_df['year_to'] == year_to)]
        pair_df = pair_df.sort_values(['source_id', 'target_id', 'area_km2'], ascending=[True, True, False])
        for row in pair_df.itertuples(index=False):
            source_id = int(row.source_id)
            target_id = int(row.target_id)
            area_val = float(row.area_km2)
            if area_val <= 0:
                continue

            left_bottom = out_cursor[year_from].get(source_id, 0.0)
            left_top = left_bottom + area_val
            right_bottom = in_cursor[year_to].get(target_id, 0.0)
            right_top = right_bottom + area_val
            out_cursor[year_from][source_id] = left_top
            in_cursor[year_to][target_id] = right_top

            polygon_points = [
                (x_positions[year_from] + bar_width, left_bottom),
                (x_positions[year_to], right_bottom),
                (x_positions[year_to], right_top),
                (x_positions[year_from] + bar_width, left_top),
            ]
            axis.add_patch(
                Polygon(
                    polygon_points,
                    closed=True,
                    facecolor=_hex_to_rgba_tuple(class_palette.get(source_id, '#999999'), alpha=0.32),
                    edgecolor='none',
                )
            )

    for year in years:
        for class_id in ids_ordered:
            bottom, top = positions[year][class_id]
            if top <= bottom:
                continue
            class_color = class_palette.get(class_id, '#999999')
            axis.add_patch(
                Rectangle(
                    (x_positions[year], bottom),
                    bar_width,
                    top - bottom,
                    facecolor=class_color,
                    edgecolor='white',
                    linewidth=0.4,
                )
            )

    first_year = years[0]
    last_year = years[-1]
    for class_id in ids_ordered:
        class_label = class_labels.get(class_id, f'Class {class_id}')
        first_bottom, first_top = positions[first_year][class_id]
        last_bottom, last_top = positions[last_year][class_id]
        if first_top > first_bottom:
            axis.text(
                x_positions[first_year] - 0.012,
                (first_bottom + first_top) / 2,
                class_label,
                ha='right',
                va='center',
                fontsize=7,
            )
        if last_top > last_bottom:
            axis.text(
                x_positions[last_year] + bar_width + 0.012,
                (last_bottom + last_top) / 2,
                class_label,
                ha='left',
                va='center',
                fontsize=7,
            )

    for year in years:
        axis.text(
            x_positions[year] + bar_width / 2,
            max_stack_height * 1.03,
            str(year),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, max_stack_height * 1.08)
    axis.axis('off')
    figure.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.985])
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return True


def _save_major_focus_alluvial(edges_df, years, class_labels, class_palette, class_ids, focus_id, output_path):
    if edges_df.empty or focus_id not in class_ids:
        return False

    transition_pairs = list(zip(years[:-1], years[1:]))
    if not transition_pairs:
        return False

    ids_present = sorted(set(edges_df['source_id']).union(set(edges_df['target_id'])))
    ids_ordered = [class_id for class_id in class_ids if class_id in ids_present]
    if focus_id not in ids_ordered:
        return False

    focus_totals = {}
    for index, year in enumerate(years):
        incoming_focus = edges_df[(edges_df['year_to'] == year) & (edges_df['target_id'] == focus_id)]['area_km2'].sum()
        outgoing_focus = edges_df[(edges_df['year_from'] == year) & (edges_df['source_id'] == focus_id)]['area_km2'].sum()
        if index == 0:
            focus_totals[year] = float(outgoing_focus)
        elif index == len(years) - 1:
            focus_totals[year] = float(incoming_focus)
        else:
            focus_totals[year] = float(max(incoming_focus, outgoing_focus))

    max_total = max(focus_totals.values()) if focus_totals else 0.0
    if max_total <= 0:
        return False

    x_start = 0.08
    x_end = 0.92
    x_positions = {
        year: x_start + idx * (x_end - x_start) / (len(years) - 1)
        for idx, year in enumerate(years)
    }
    bar_width = 0.028

    focus_positions = {year: (0.0, focus_totals[year]) for year in years}

    out_cursor = {year: 0.0 for year in years}
    in_cursor = {year: 0.0 for year in years}

    focus_color = class_palette.get(focus_id, '#67179C')
    retained_color = focus_color

    figure_width = float(TRANSITION_CONFIG.get('multiyear_flow_width_in', 6.5))
    figure, axis = plt.subplots(figsize=(figure_width, 4.8))

    y_extent_max = max_total
    anchor_gap = max(max_total * 0.015, 1.0)

    for year_from, year_to in transition_pairs:
        pair_df = edges_df[(edges_df['year_from'] == year_from) & (edges_df['year_to'] == year_to)]
        pair_df = pair_df.sort_values(['source_id', 'target_id', 'area_km2'], ascending=[True, True, False])

        retained_rows = pair_df[(pair_df['source_id'] == focus_id) & (pair_df['target_id'] == focus_id)]
        retained_val = float(retained_rows['area_km2'].sum())

        out_rows = pair_df[(pair_df['source_id'] == focus_id) & (pair_df['target_id'] != focus_id)].copy()
        out_rows = out_rows.sort_values('target_id')

        in_rows = pair_df[(pair_df['source_id'] != focus_id) & (pair_df['target_id'] == focus_id)].copy()
        in_rows = in_rows.sort_values('source_id')

        left_retained_bottom = out_cursor[year_from]
        left_retained_top = left_retained_bottom + retained_val
        out_cursor[year_from] = left_retained_top

        right_retained_bottom = in_cursor[year_to]
        right_retained_top = right_retained_bottom + retained_val
        in_cursor[year_to] = right_retained_top

        if retained_val > 0:
            retained_poly = [
                (x_positions[year_from] + bar_width, left_retained_bottom),
                (x_positions[year_to], right_retained_bottom),
                (x_positions[year_to], right_retained_top),
                (x_positions[year_from] + bar_width, left_retained_top),
            ]
            axis.add_patch(
                Polygon(
                    retained_poly,
                    closed=True,
                    facecolor=_hex_to_rgba_tuple(retained_color, alpha=0.52),
                    edgecolor='none',
                )
            )

        pair_anchor_base = max_total * 1.05
        pair_dx = x_positions[year_to] - x_positions[year_from]
        out_anchor_x = x_positions[year_from] + 0.72 * pair_dx
        in_anchor_x = x_positions[year_from] + 0.28 * pair_dx

        out_anchor_cursor = pair_anchor_base
        for row in out_rows.itertuples(index=False):
            area_val = float(row.area_km2)
            if area_val <= 0:
                continue
            target_id = int(row.target_id)
            class_color = class_palette.get(target_id, '#999999')

            left_bottom = out_cursor[year_from]
            left_top = left_bottom + area_val
            out_cursor[year_from] = left_top

            anchor_bottom = out_anchor_cursor
            anchor_top = anchor_bottom + area_val
            out_anchor_cursor = anchor_top + anchor_gap
            y_extent_max = max(y_extent_max, anchor_top)

            out_poly = [
                (x_positions[year_from] + bar_width, left_bottom),
                (out_anchor_x, anchor_bottom),
                (out_anchor_x, anchor_top),
                (x_positions[year_from] + bar_width, left_top),
            ]
            axis.add_patch(
                Polygon(
                    out_poly,
                    closed=True,
                    facecolor=_hex_to_rgba_tuple(class_color, alpha=0.48),
                    edgecolor='none',
                )
            )

        in_anchor_cursor = pair_anchor_base
        for row in in_rows.itertuples(index=False):
            area_val = float(row.area_km2)
            if area_val <= 0:
                continue
            source_id = int(row.source_id)
            class_color = class_palette.get(source_id, '#999999')

            right_bottom = in_cursor[year_to]
            right_top = right_bottom + area_val
            in_cursor[year_to] = right_top

            anchor_bottom = in_anchor_cursor
            anchor_top = anchor_bottom + area_val
            in_anchor_cursor = anchor_top + anchor_gap
            y_extent_max = max(y_extent_max, anchor_top)

            in_poly = [
                (in_anchor_x, anchor_bottom),
                (x_positions[year_to], right_bottom),
                (x_positions[year_to], right_top),
                (in_anchor_x, anchor_top),
            ]
            axis.add_patch(
                Polygon(
                    in_poly,
                    closed=True,
                    facecolor=_hex_to_rgba_tuple(class_color, alpha=0.48),
                    edgecolor='none',
                )
            )

    for year in years:
        bottom, top = focus_positions[year]
        if top <= bottom:
            continue
        axis.add_patch(
            Rectangle(
                (x_positions[year], bottom),
                bar_width,
                top - bottom,
                facecolor=focus_color,
                edgecolor='white',
                linewidth=0.6,
            )
        )

    focus_label = class_labels.get(focus_id, f'Class {focus_id}')
    first_year = years[0]
    last_year = years[-1]
    first_bottom, first_top = focus_positions[first_year]
    last_bottom, last_top = focus_positions[last_year]
    first_total = max(first_top - first_bottom, 0.0)
    last_total = max(last_top - last_bottom, 0.0)
    left_label = f"{focus_label} ({first_total:,.0f} km²)"
    right_label = f"{focus_label} ({last_total:,.0f} km²)"
    axis.text(x_positions[first_year] - 0.012, (first_bottom + first_top) / 2, left_label, ha='right', va='center', fontsize=8, fontweight='bold')
    axis.text(x_positions[last_year] + bar_width + 0.012, (last_bottom + last_top) / 2, right_label, ha='left', va='center', fontsize=8, fontweight='bold')

    for year in years:
        axis.text(x_positions[year] + bar_width / 2, y_extent_max * 1.03, str(year), ha='center', va='bottom', fontsize=9, fontweight='bold')

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=_hex_to_rgba_tuple(retained_color, 0.50), edgecolor='none')
    ]
    legend_labels = [f"{focus_label} (retained)"]

    class_legend_ids = [class_id for class_id in ids_ordered if class_id != focus_id]
    for class_id in class_legend_ids:
        class_color = class_palette.get(class_id, '#999999')
        legend_handles.append(
            Rectangle((0, 0), 1, 1, facecolor=_hex_to_rgba_tuple(class_color, 0.50), edgecolor='none')
        )
        legend_labels.append(class_labels.get(class_id, f'Class {class_id}'))

    axis.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, max(1, len(legend_labels))),
        frameon=False,
        fontsize=8,
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, y_extent_max * 1.10)
    axis.axis('off')
    figure.suptitle(f'{focus_label}: Yearly Total with Inflow/Outflow (km²)', fontsize=12, fontweight='bold', y=0.995)
    figure.tight_layout(rect=[0, 0.02, 1, 0.98])
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return True


def _build_major_edges(edges_df):
    if edges_df.empty:
        return pd.DataFrame()

    major_df = (
        edges_df
        .groupby(['year_from', 'year_to', 'year_pair', 'source_type', 'target_type'], as_index=False)['area_km2']
        .sum()
    )
    source_totals = major_df.groupby(['year_from', 'year_to', 'source_type'])['area_km2'].sum().to_dict()
    major_df['pct_of_source'] = major_df.apply(
        lambda row: (row['area_km2'] / source_totals.get((row['year_from'], row['year_to'], row['source_type']), 1.0)) * 100
        if source_totals.get((row['year_from'], row['year_to'], row['source_type']), 0) > 0
        else 0.0,
        axis=1,
    )
    return major_df


def _build_type_palette(class_to_type, class_palette):
    palette = {}
    for class_id, type_name in class_to_type.items():
        if type_name not in palette and class_id in class_palette:
            palette[type_name] = class_palette[class_id]
    return palette


schema = TRANSITION_CONFIG.get('transition_schema', 'detailed').lower()
if schema not in {'detailed', 'aggregated'}:
    raise ValueError("transition_schema must be 'detailed' or 'aggregated'")

class_aggregation = _int_key_dict(TRANSITION_CONFIG.get('class_aggregation', CONSOL_CONFIG.get('class_aggregation', {})))
class_labels, class_palette, class_to_type, class_ids = _schema_settings(schema)

analysis_years = sorted({int(year) for year in TRANSITION_CONFIG.get('years', CONSOL_CONFIG.get('years', []))})
if len(analysis_years) < 2:
    raise ValueError('Transition analysis needs at least two years.')

asset_folder = TRANSITION_CONFIG.get('asset_folder', EMB_CONFIG.get('asset_folder', f"{config.CURRENT_STUDY_AREA}/classification"))
asset_prefix = TRANSITION_CONFIG.get('asset_prefix', CONSOL_CONFIG.get('asset_prefix', 'classified'))
scale = int(TRANSITION_CONFIG.get('scale', 30))
min_flow_km2 = float(TRANSITION_CONFIG.get('min_flow_km2', 50.0))
multiplier = max(class_ids) + 1

print(f"Study area: {study_area_name}")
print(f"Schema: {schema}")
print(f"Years requested: {analysis_years}")
print(f"Asset source: projects/{config.PROJECT_ID}/assets/{asset_folder}/{asset_prefix}_YYYY")

available_years = []
missing_years = []
for year in analysis_years:
    asset_id = f"projects/{config.PROJECT_ID}/assets/{asset_folder}/{asset_prefix}_{year}"
    if asset_exists(asset_id):
        available_years.append(year)
        print(f"  {year}: ✓ Found")
    else:
        missing_years.append(year)
        print(f"  {year}: ✗ Missing")

if len(available_years) < 2:
    raise RuntimeError('Not enough available yearly assets to compute transitions.')

transition_pairs = list(zip(available_years[:-1], available_years[1:]))
print(f"Transition pairs: {transition_pairs}")

all_edges = []
pair_summaries = []

for year_from, year_to in transition_pairs:
    pair_name = f"{year_from}_{year_to}"
    asset_from = f"projects/{config.PROJECT_ID}/assets/{asset_folder}/{asset_prefix}_{year_from}"
    asset_to = f"projects/{config.PROJECT_ID}/assets/{asset_folder}/{asset_prefix}_{year_to}"

    print(f"\nComputing transitions {year_from} → {year_to}...")
    image_from = _prepare_class_image(asset_from, schema, class_aggregation)
    image_to = _prepare_class_image(asset_to, schema, class_aggregation)
    grouped = _compute_transition_groups(image_from, image_to, study_area, scale, multiplier)

    pair_records = []
    for group in grouped:
        transition_code = int(group['transition'])
        area_km2 = float(group['sum'])
        source_id = transition_code // multiplier
        target_id = transition_code % multiplier
        source_label = class_labels.get(source_id, f'Class {source_id}')
        target_label = class_labels.get(target_id, f'Class {target_id}')
        source_type = class_to_type.get(source_id, 'Other')
        target_type = class_to_type.get(target_id, 'Other')

        pair_records.append(
            {
                'year_from': int(year_from),
                'year_to': int(year_to),
                'year_pair': pair_name,
                'source_id': int(source_id),
                'source_label': source_label,
                'source_type': source_type,
                'target_id': int(target_id),
                'target_label': target_label,
                'target_type': target_type,
                'area_km2': area_km2,
            }
        )

    if not pair_records:
        print('  No transition records found for this pair.')
        continue

    pair_df = pd.DataFrame(pair_records)
    source_totals = pair_df.groupby('source_id')['area_km2'].sum().to_dict()
    pair_df['pct_of_source'] = pair_df.apply(
        lambda row: (row['area_km2'] / source_totals.get(row['source_id'], 1.0)) * 100
        if source_totals.get(row['source_id'], 0) > 0
        else 0.0,
        axis=1,
    )

    matrix = (
        pair_df
        .pivot_table(index='source_label', columns='target_label', values='area_km2', aggfunc='sum', fill_value=0)
    )

    ordered_labels = [class_labels[class_id] for class_id in class_ids if class_id in class_labels]
    matrix = matrix.reindex(index=ordered_labels, columns=ordered_labels, fill_value=0)

    matrix_path = MATRICES_DIR / f'transition_matrix_{pair_name}.csv'
    matrix.to_csv(matrix_path)
    print(f"  ✓ Matrix saved: {matrix_path}")

    pair_edges_path = OUTPUT_DIR / f'transition_edges_{pair_name}.csv'
    pair_df.sort_values(['source_id', 'target_id']).to_csv(pair_edges_path, index=False)
    print(f"  ✓ Pair edges saved: {pair_edges_path}")

    pair_summaries.append(
        {
            'year_pair': pair_name,
            'year_from': int(year_from),
            'year_to': int(year_to),
            'n_edges': int(len(pair_df)),
            'total_area_km2': float(pair_df['area_km2'].sum()),
            'matrix_csv': str(matrix_path),
            'edges_csv': str(pair_edges_path),
        }
    )

    all_edges.append(pair_df)

if not all_edges:
    raise RuntimeError('No transition edges computed across available pairs.')

all_edges_df = pd.concat(all_edges, ignore_index=True)
all_edges_path = OUTPUT_DIR / 'transition_edges_all_pairs.csv'
all_edges_df.sort_values(['year_from', 'source_id', 'target_id']).to_csv(all_edges_path, index=False)
print(f"\n✓ All pair edges saved: {all_edges_path}")

filtered_edges_df = all_edges_df[all_edges_df['area_km2'] >= min_flow_km2].copy()
filtered_edges_path = OUTPUT_DIR / f'transition_edges_filtered_min_{min_flow_km2:g}_km2.csv'
filtered_edges_df.sort_values(['year_from', 'source_id', 'target_id']).to_csv(filtered_edges_path, index=False)
print(f"✓ Filtered edges saved: {filtered_edges_path}")

major_edges_df = _build_major_edges(all_edges_df)
major_edges_path = OUTPUT_DIR / 'transition_edges_major_all_pairs.csv'
major_edges_df.sort_values(['year_from', 'source_type', 'target_type']).to_csv(major_edges_path, index=False)
print(f"✓ Major-class edges saved: {major_edges_path}")

major_filtered_df = major_edges_df[major_edges_df['area_km2'] >= min_flow_km2].copy()
major_filtered_path = OUTPUT_DIR / f'transition_edges_major_filtered_min_{min_flow_km2:g}_km2.csv'
major_filtered_df.sort_values(['year_from', 'source_type', 'target_type']).to_csv(major_filtered_path, index=False)
print(f"✓ Major-class filtered edges saved: {major_filtered_path}")

major_types = sorted(set(major_edges_df['source_type']).union(set(major_edges_df['target_type']))) if not major_edges_df.empty else []
major_type_to_id = {type_name: idx for idx, type_name in enumerate(major_types)}
major_id_to_type = {idx: type_name for type_name, idx in major_type_to_id.items()}
major_type_palette = _build_type_palette(class_to_type, class_palette)
major_palette_by_id = {major_type_to_id[type_name]: major_type_palette.get(type_name, '#999999') for type_name in major_types}
major_labels_by_id = {major_type_to_id[type_name]: type_name for type_name in major_types}

if not major_edges_df.empty:
    major_plot_df = major_edges_df.copy()
    major_plot_df['source_id'] = major_plot_df['source_type'].map(major_type_to_id)
    major_plot_df['target_id'] = major_plot_df['target_type'].map(major_type_to_id)
    major_plot_df['source_label'] = major_plot_df['source_type']
    major_plot_df['target_label'] = major_plot_df['target_type']

    for year_from, year_to in transition_pairs:
        pair_name = f"{year_from}_{year_to}"
        pair_major = major_edges_df[(major_edges_df['year_from'] == year_from) & (major_edges_df['year_to'] == year_to)]
        if pair_major.empty:
            continue
        matrix_major = (
            pair_major
            .pivot_table(index='source_type', columns='target_type', values='area_km2', aggfunc='sum', fill_value=0)
            .reindex(index=major_types, columns=major_types, fill_value=0)
        )
        matrix_major_path = MATRICES_MAJOR_DIR / f'transition_matrix_major_{pair_name}.csv'
        matrix_major.to_csv(matrix_major_path)
        print(f"  ✓ Major matrix saved: {matrix_major_path}")

        pair_major_path = OUTPUT_DIR / f'transition_edges_major_{pair_name}.csv'
        pair_major.sort_values(['source_type', 'target_type']).to_csv(pair_major_path, index=False)
        print(f"  ✓ Major pair edges saved: {pair_major_path}")

else:
    major_plot_df = pd.DataFrame()

alluvial_outputs = []
alluvial_major_outputs = []
alluvial_major_focus_outputs = []

if TRANSITION_CONFIG.get('generate_pairwise_flow_plots', TRANSITION_CONFIG.get('generate_pairwise_sankey', True)):
    for year_from, year_to in transition_pairs:
        pair_df = filtered_edges_df[(filtered_edges_df['year_from'] == year_from) & (filtered_edges_df['year_to'] == year_to)]
        if pair_df.empty:
            continue
        alluvial_path = OUTPUT_DIR / f'alluvial_{year_from}_{year_to}.png'
        created = _save_pairwise_alluvial(pair_df, year_from, year_to, class_labels, class_palette, class_ids, alluvial_path)
        if created:
            alluvial_outputs.append(str(alluvial_path))
            print(f"✓ Alluvial plot saved: {alluvial_path}")

    if not major_plot_df.empty:
        major_filtered_plot_df = major_plot_df[major_plot_df['area_km2'] >= min_flow_km2].copy()
        for year_from, year_to in transition_pairs:
            pair_major = major_filtered_plot_df[(major_filtered_plot_df['year_from'] == year_from) & (major_filtered_plot_df['year_to'] == year_to)]
            if pair_major.empty:
                continue
            major_alluvial_path = OUTPUT_DIR / f'alluvial_major_{year_from}_{year_to}.png'
            created_major = _save_pairwise_alluvial(
                pair_major,
                year_from,
                year_to,
                major_labels_by_id,
                major_palette_by_id,
                sorted(major_labels_by_id.keys()),
                major_alluvial_path,
            )
            if created_major:
                alluvial_major_outputs.append(str(major_alluvial_path))
                print(f"✓ Major-class alluvial plot saved: {major_alluvial_path}")

if TRANSITION_CONFIG.get('generate_multiyear_flow_plot', TRANSITION_CONFIG.get('generate_multiyear_sankey', True)) and not filtered_edges_df.empty:
    alluvial_sequence_path = OUTPUT_DIR / f"alluvial_left_to_right_{available_years[0]}_{available_years[-1]}.png"
    created = _save_left_to_right_alluvial(
        filtered_edges_df,
        available_years,
        class_labels,
        class_palette,
        class_ids,
        alluvial_sequence_path,
        title='Class Transition Flow Across All Years (km²)',
    )
    if created:
        alluvial_outputs.append(str(alluvial_sequence_path))
        print(f"✓ Left-to-right all-years alluvial saved: {alluvial_sequence_path}")

    if not major_plot_df.empty:
        major_filtered_plot_df = major_plot_df[major_plot_df['area_km2'] >= min_flow_km2].copy()
        if not major_filtered_plot_df.empty:
            major_lr_path = OUTPUT_DIR / f"alluvial_major_left_to_right_{available_years[0]}_{available_years[-1]}.png"
            created_major_lr = _save_left_to_right_alluvial(
                major_filtered_plot_df,
                available_years,
                major_labels_by_id,
                major_palette_by_id,
                sorted(major_labels_by_id.keys()),
                major_lr_path,
                title='Major Landcover Class Transition Flow Across All Years (km²)',
            )
            if created_major_lr:
                alluvial_major_outputs.append(str(major_lr_path))
                print(f"✓ Major-class left-to-right alluvial saved: {major_lr_path}")

if TRANSITION_CONFIG.get('generate_major_focus_flow_plots', True) and not major_plot_df.empty:
    major_filtered_plot_df = major_plot_df[major_plot_df['area_km2'] >= min_flow_km2].copy()
    focus_category_names = TRANSITION_CONFIG.get('major_focus_categories', [])
    if focus_category_names:
        focus_type_names = [name for name in focus_category_names if name in major_type_to_id]
    else:
        focus_type_names = major_types

    for focus_name in focus_type_names:
        focus_id = major_type_to_id[focus_name]
        focus_filename = focus_name.lower().replace(' ', '_').replace('.', '').replace('/', '_')
        focus_path = OUTPUT_DIR / f'alluvial_major_focus_{focus_filename}_{available_years[0]}_{available_years[-1]}.png'
        created_focus = _save_major_focus_alluvial(
            major_filtered_plot_df,
            available_years,
            major_labels_by_id,
            major_palette_by_id,
            sorted(major_labels_by_id.keys()),
            focus_id,
            focus_path,
        )
        if created_focus:
            alluvial_major_focus_outputs.append(str(focus_path))
            print(f"✓ Major-class focus flow saved: {focus_path}")


# Consolidated early/late transition using confidence/probability weights
consolidated_transition_outputs = {}
if TRANSITION_CONFIG.get('compute_consolidated_early_late_transition', True):
    cons_asset_folder = CONSOL_CONFIG.get('asset_folder', asset_folder)
    cons_early_name = TRANSITION_CONFIG.get('consolidated_early_asset_name', CONSOL_CONFIG.get('early_asset_name', 'consolidated_2024_weighted'))
    cons_late_name = TRANSITION_CONFIG.get('consolidated_late_asset_name', CONSOL_CONFIG.get('late_asset_name', 'consolidated_2019_weighted'))
    cons_early_asset = f"projects/{config.PROJECT_ID}/assets/{cons_asset_folder}/{cons_early_name}"
    cons_late_asset = f"projects/{config.PROJECT_ID}/assets/{cons_asset_folder}/{cons_late_name}"

    print("\nComputing consolidated early↔late transition...")
    if not asset_exists(cons_early_asset):
        print(f"  ⚠️ Missing consolidated early asset: {cons_early_asset}")
    elif not asset_exists(cons_late_asset):
        print(f"  ⚠️ Missing consolidated late asset: {cons_late_asset}")
    else:
        cons_labels = _int_key_dict(CONSOL_CONFIG.get('aggregated_labels', {}))
        cons_palette = _int_key_dict(CONSOL_CONFIG.get('aggregated_palette', {}))
        cons_ids = sorted(cons_labels.keys())
        cons_mult = max(cons_ids) + 1 if cons_ids else 20

        early_image = ee.Image(cons_early_asset)
        late_image = ee.Image(cons_late_asset)
        early_bands = set(_retry_with_backoff(lambda: early_image.bandNames().getInfo()))
        late_bands = set(_retry_with_backoff(lambda: late_image.bandNames().getInfo()))

        early_class = early_image.select('classification').toInt16()
        late_class = late_image.select('classification').toInt16()

        valid_mask = early_class.gte(0).And(late_class.gte(0))
        confidence_weight = ee.Image.constant(1).toFloat()
        use_confidence = TRANSITION_CONFIG.get('consolidated_use_confidence_weight', True)
        min_conf = float(TRANSITION_CONFIG.get('consolidated_min_confidence', 0))

        if use_confidence and ('confidence' in early_bands) and ('confidence' in late_bands):
            early_conf = early_image.select('confidence').toFloat()
            late_conf = late_image.select('confidence').toFloat()
            valid_mask = valid_mask.And(early_conf.gte(min_conf)).And(late_conf.gte(min_conf))
            confidence_weight = early_conf.divide(100.0).multiply(late_conf.divide(100.0)).rename('confidence_weight')

        probability_weight = ee.Image.constant(1).toFloat()
        use_probability = TRANSITION_CONFIG.get('consolidated_use_probability_weight', True)
        if use_probability and cons_ids:
            band_by_class = {class_id: _probability_band_name(cons_labels[class_id]) for class_id in cons_ids if class_id in cons_labels}
            probs_available = all((band_by_class[class_id] in early_bands and band_by_class[class_id] in late_bands) for class_id in band_by_class)
            if probs_available and band_by_class:
                early_prob_selected = _select_class_probability(early_image, early_class, cons_ids, band_by_class)
                late_prob_selected = _select_class_probability(late_image, late_class, cons_ids, band_by_class)
                probability_weight = early_prob_selected.multiply(late_prob_selected).rename('probability_weight')
            else:
                print("  ⚠️ Probability bands missing for one or more classes; using confidence-only weighting.")

        transition_code = early_class.multiply(cons_mult).add(late_class).toInt32().rename('transition')
        transition_code = transition_code.updateMask(valid_mask)

        pixel_area_km2 = ee.Image.pixelArea().divide(1e6).rename('area_km2')
        weighted_factor = confidence_weight.multiply(probability_weight).rename('transition_weight')
        weighted_area = pixel_area_km2.multiply(weighted_factor).rename('area_km2')

        raw_groups = _compute_transition_groups_with_area(transition_code, pixel_area_km2.updateMask(valid_mask), study_area, scale)
        weighted_groups = _compute_transition_groups_with_area(transition_code, weighted_area.updateMask(valid_mask), study_area, scale)

        raw_map = {int(group['transition']): float(group['sum']) for group in raw_groups}
        weighted_map = {int(group['transition']): float(group['sum']) for group in weighted_groups}

        records = []
        for transition_code_val in sorted(set(raw_map.keys()).union(set(weighted_map.keys()))):
            source_id = int(transition_code_val // cons_mult)
            target_id = int(transition_code_val % cons_mult)
            raw_area_val = float(raw_map.get(transition_code_val, 0.0))
            weighted_area_val = float(weighted_map.get(transition_code_val, 0.0))
            if raw_area_val <= 0 and weighted_area_val <= 0:
                continue
            records.append(
                {
                    'source_id': source_id,
                    'source_label': cons_labels.get(source_id, f'Class {source_id}'),
                    'target_id': target_id,
                    'target_label': cons_labels.get(target_id, f'Class {target_id}'),
                    'raw_area_km2': raw_area_val,
                    'weighted_area_km2': weighted_area_val,
                    'weight_to_raw_pct': (weighted_area_val / raw_area_val * 100) if raw_area_val > 0 else 0.0,
                }
            )

        if records:
            cons_df = pd.DataFrame(records)
            cons_df = cons_df.sort_values(['source_id', 'target_id']).reset_index(drop=True)

            cons_edges_path = OUTPUT_DIR / 'transition_edges_consolidated_early_late.csv'
            cons_df.to_csv(cons_edges_path, index=False)
            print(f"  ✓ Consolidated transition edges saved: {cons_edges_path}")

            cons_matrix_raw = (
                cons_df
                .pivot_table(index='source_label', columns='target_label', values='raw_area_km2', aggfunc='sum', fill_value=0)
                .reindex(index=[cons_labels[class_id] for class_id in cons_ids], columns=[cons_labels[class_id] for class_id in cons_ids], fill_value=0)
            )
            cons_matrix_weighted = (
                cons_df
                .pivot_table(index='source_label', columns='target_label', values='weighted_area_km2', aggfunc='sum', fill_value=0)
                .reindex(index=[cons_labels[class_id] for class_id in cons_ids], columns=[cons_labels[class_id] for class_id in cons_ids], fill_value=0)
            )

            cons_matrix_raw_path = OUTPUT_DIR / 'transition_matrix_consolidated_early_late_raw.csv'
            cons_matrix_weighted_path = OUTPUT_DIR / 'transition_matrix_consolidated_early_late_weighted.csv'
            cons_matrix_raw.to_csv(cons_matrix_raw_path)
            cons_matrix_weighted.to_csv(cons_matrix_weighted_path)
            print(f"  ✓ Consolidated raw matrix saved: {cons_matrix_raw_path}")
            print(f"  ✓ Consolidated weighted matrix saved: {cons_matrix_weighted_path}")

            cons_plot_df = cons_df.rename(columns={'raw_area_km2': 'area_km2'}).copy()
            cons_plot_df['area_km2'] = cons_df['weighted_area_km2']
            cons_plot_df = cons_plot_df[cons_plot_df['area_km2'] >= min_flow_km2]
            cons_plot_df['year_from'] = int(min(available_years))
            cons_plot_df['year_to'] = int(max(available_years))

            cons_alluvial_path = OUTPUT_DIR / 'alluvial_consolidated_early_late_weighted.png'
            if not cons_plot_df.empty:
                created_cons_plot = _save_pairwise_alluvial(
                    cons_plot_df,
                    year_from='Early',
                    year_to='Late',
                    class_labels=cons_labels,
                    class_palette=cons_palette,
                    class_ids=cons_ids,
                    output_path=cons_alluvial_path,
                    figure_width=4.5,
                    include_node_totals=True,
                )
                if created_cons_plot:
                    print(f"  ✓ Consolidated weighted alluvial saved: {cons_alluvial_path}")

            consolidated_transition_outputs = {
                'early_asset': cons_early_asset,
                'late_asset': cons_late_asset,
                'edges_csv': str(cons_edges_path),
                'raw_matrix_csv': str(cons_matrix_raw_path),
                'weighted_matrix_csv': str(cons_matrix_weighted_path),
                'alluvial_png': str(cons_alluvial_path) if cons_plot_df is not None and not cons_plot_df.empty else None,
                'use_confidence_weight': bool(use_confidence),
                'use_probability_weight': bool(use_probability),
                'min_confidence': min_conf,
            }

summary = {
    'project_id': config.PROJECT_ID,
    'study_area': study_area_name,
    'date_processed': pd.Timestamp.now().isoformat(),
    'schema': schema,
    'scale_meters': scale,
    'min_flow_km2': min_flow_km2,
    'years_requested': analysis_years,
    'years_available': available_years,
    'years_missing': missing_years,
    'pairs_processed': [f"{year_from}_{year_to}" for year_from, year_to in transition_pairs],
    'class_labels': {str(class_id): class_labels[class_id] for class_id in class_ids if class_id in class_labels},
    'outputs': {
        'all_edges_csv': str(all_edges_path),
        'filtered_edges_csv': str(filtered_edges_path),
        'matrices_dir': str(MATRICES_DIR),
        'alluvial_png': alluvial_outputs,
        'major_edges_csv': str(major_edges_path),
        'major_filtered_edges_csv': str(major_filtered_path),
        'major_matrices_dir': str(MATRICES_MAJOR_DIR),
        'major_alluvial_png': alluvial_major_outputs,
        'major_focus_alluvial_png': alluvial_major_focus_outputs,
        'consolidated_early_late': consolidated_transition_outputs,
    },
    'pair_summaries': pair_summaries,
}

summary_path = OUTPUT_DIR / 'transition_summary.json'
with open(summary_path, 'w') as file:
    json.dump(summary, file, indent=2)
print(f"✓ Summary saved: {summary_path}")

print('\n' + '=' * 70)
print('TRANSITION ANALYSIS COMPLETE')
print('=' * 70)
print(f"Study area: {study_area_name}")
print(f"Schema: {schema}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Transition pairs processed: {[f'{year_from}->{year_to}' for year_from, year_to in transition_pairs]}")
print('=' * 70)

# %%
