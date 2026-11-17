
import numpy as np

from src.build_feature_sets import (
    compute_s0_unmasked_rgb_means,
    compute_s4_rgb_hsv_statistics,
    encode_device_lamp,
    build_feature_registry,
)


def test_s0_includes_transparent_pixels_as_black():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)

    rgba[0, 0] = [255, 0, 0, 255]
    rgba[0, 1] = [0, 255, 0, 255]
    rgba[1, 0] = [0, 0, 255, 0]
    rgba[1, 1] = [255, 255, 255, 0]

    result = compute_s0_unmasked_rgb_means(rgba)

    assert result["S0_R_mean"] == 63.75
    assert result["S0_G_mean"] == 63.75
    assert result["S0_B_mean"] == 0.0


def test_s4_uses_only_cleaned_roi_pixels():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)

    rgba[:, :] = [255, 0, 0, 0]
    rgba[2:8, 2:8] = [0, 0, 255, 255]

    result = compute_s4_rgb_hsv_statistics(rgba)

    assert result["S4_rgb_R_mean"] == 0.0
    assert result["S4_rgb_G_mean"] == 0.0
    assert result["S4_rgb_B_mean"] == 255.0

    assert result["S4_hsv_S_mean"] == 1.0
    assert result["S4_hsv_V_mean"] == 1.0


def test_device_lamp_one_hot():
    dino = encode_device_lamp("dino", "off")

    assert dino == {
        "device_dino": 1,
        "device_phone": 0,
        "lamp_off": 1,
        "lamp_on": 0,
    }

    phone = encode_device_lamp("phone", "on")

    assert phone == {
        "device_dino": 0,
        "device_phone": 1,
        "lamp_off": 0,
        "lamp_on": 1,
    }


def test_feature_registry_has_five_nested_sets():
    columns = {
        "S0_R_mean", "S0_G_mean", "S0_B_mean",
        "L_mean", "a_mean", "b_mean",
        "L_med", "a_med", "b_med",
        "L_std", "a_std", "b_std",
        "L_p10", "a_p10", "b_p10",
        "L_p90", "a_p90", "b_p90",
        "pale_fraction",
        "chroma_mean", "chroma_med", "chroma_std",
        "chroma_p10", "chroma_p90",
        "hue_angle_deg",
    }

    for colour_space, channels in (
        ("rgb", ("R", "G", "B")),
        ("hsv", ("H", "S", "V")),
    ):
        for channel in channels:
            prefix = f"S4_{colour_space}_{channel}"

            for statistic in ("mean", "med", "std", "p10", "p90"):
                columns.add(f"{prefix}_{statistic}")

    columns.add("S4_hsv_H_circular_mean")

    registry = build_feature_registry(columns)

    assert list(registry) == ["S0", "S1", "S2", "S3", "S4"]

    assert set(registry["S1"]["colour_only"]).issubset(
        registry["S2"]["colour_only"]
    )

    assert set(registry["S2"]["colour_only"]).issubset(
        registry["S3"]["colour_only"]
    )

    assert set(registry["S3"]["colour_only"]).issubset(
        registry["S4"]["colour_only"]
    )

    for variants in registry.values():
        for variant_columns in variants.values():
            assert not any(
                column.startswith("qc_")
                for column in variant_columns
            )

            assert "conc" not in variant_columns
            assert "tube_id" not in variant_columns
            assert "role" not in variant_columns


def test_each_set_has_device_lamp_variant():
    columns = {
        "S0_R_mean", "S0_G_mean", "S0_B_mean",
        "L_mean", "a_mean", "b_mean",
        "L_med", "a_med", "b_med",
        "L_std", "a_std", "b_std",
        "L_p10", "a_p10", "b_p10",
        "L_p90", "a_p90", "b_p90",
        "pale_fraction",
        "chroma_mean", "chroma_med", "chroma_std",
        "chroma_p10", "chroma_p90",
        "hue_angle_deg",
    }

    for colour_space, channels in (
        ("rgb", ("R", "G", "B")),
        ("hsv", ("H", "S", "V")),
    ):
        for channel in channels:
            prefix = f"S4_{colour_space}_{channel}"

            for statistic in ("mean", "med", "std", "p10", "p90"):
                columns.add(f"{prefix}_{statistic}")

    columns.add("S4_hsv_H_circular_mean")

    registry = build_feature_registry(columns)

    for variants in registry.values():
        colour = variants["colour_only"]
        with_device = variants["with_device_and_lamp"]

        assert len(with_device) == len(colour) + 4
        assert with_device[-4:] == [
            "device_dino",
            "device_phone",
            "lamp_off",
            "lamp_on",
        ]
