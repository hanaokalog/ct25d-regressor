"""Physical constants of the preprocessing grid.

Changing these changes what the trained weights mean, so they live in one
place and are imported everywhere rather than repeated as defaults.
"""

TARGET_INPLANE_MM = 0.78125          # resampled in-plane pixel size
SLICE_GAP_MM = 5.0                   # through-plane distance between slices
SLAB_MM = 5.0                        # each plane is the mean over this thickness
PIXEL_AREA_MM2 = TARGET_INPLANE_MM ** 2      # 0.610352 mm^2
CT_WINDOW = (-1000.0, 1000.0)        # air through cortical bone; fat stays visible
AIR_HU = -1024.0                     # value used outside the field of view

__all__ = ["TARGET_INPLANE_MM", "SLICE_GAP_MM", "SLAB_MM", "PIXEL_AREA_MM2",
           "CT_WINDOW", "AIR_HU"]
