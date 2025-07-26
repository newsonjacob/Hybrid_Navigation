import numpy as np
from uav.perception import PerceptionData, FrameStats


def test_frame_stats_attributes():
    data = PerceptionData(
        vis_img=np.zeros((1, 1, 3), dtype=np.uint8),
        good_old=np.zeros((0, 2)),
        flow_vectors=np.zeros((0, 2)),
        flow_std=0.0,
        simgetimage_s=0.0,
        decode_s=0.0,
        processing_s=0.0,
    )
    stats = FrameStats(
        smooth_L=1.0,
        smooth_C=2.0,
        smooth_R=3.0,
        delta_L=0.1,
        delta_C=0.2,
        delta_R=0.3,
        probe_mag=0.4,
        probe_count=5,
        left_count=1,
        center_count=2,
        right_count=3,
        top_mag=0.5,
        mid_mag=0.6,
        bottom_mag=0.7,
        top_count=4,
        mid_count=5,
        bottom_count=6,
        in_grace=True,
    )
    assert stats.smooth_L == 1.0
    assert stats.probe_count == 5
    assert isinstance(data, PerceptionData)

