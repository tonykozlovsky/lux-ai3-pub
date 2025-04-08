import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExtractor(nn.Module):
    def __init__(self, patch_size=15, patches_per_batch=16):
        super().__init__()
        self.n = patch_size      # size of each patch
        self.M = patches_per_batch

    def slices(self, x, units_masks):
        batch_size, num_channels, map_height, map_width = x.shape
        n = self.n
        M = self.M
        H, W = map_height, map_width

        # Coordinates are shape [B, M]
        x_cord = units_masks['x_cord']
        y_cord = units_masks['y_cord']

        # Repeat each batch item M times
        # Result: [B*M, C, H, W]
        fm_expanded = x.unsqueeze(1).expand(-1, M, -1, -1, -1)
        fm_expanded = fm_expanded.reshape(batch_size * M, num_channels, H, W)

        # Flatten the coords in the same order: shape [B*M]
        xs_flat = x_cord.reshape(-1)
        ys_flat = y_cord.reshape(-1)

        # Build the base grid from [-1..1], size n×n
        linspace_1d = torch.linspace(-1, 1, steps=n, device=x.device)
        grid_y, grid_x = torch.meshgrid(linspace_1d, linspace_1d, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)         # (n, n, 2)
        base_grid = base_grid.unsqueeze(0).expand(batch_size * M, n, n, 2)

        # Convert (x,y) from [0..W-1]/[0..H-1] -> [-1..1]
        # so pixel i => normalized => (i/(W-1))*2 -1
        xs_norm = (xs_flat / (W - 1)) * 2 - 1
        ys_norm = (ys_flat / (H - 1)) * 2 - 1
        xs_norm = xs_norm.view(-1, 1, 1)
        ys_norm = ys_norm.view(-1, 1, 1)

        # Use (n-1)/(W-1), ensures the patch spans exactly +/- 7 pixels from the center
        scale_x = (n - 1) / (W - 1)  # e.g. 14/23 ~ 0.6087
        scale_y = (n - 1) / (H - 1)  # e.g. 14/23 ~ 0.6087

        final_grid = torch.zeros_like(base_grid)
        final_grid[..., 0] = base_grid[..., 0] * scale_x + ys_norm
        final_grid[..., 1] = base_grid[..., 1] * scale_y + xs_norm

        # For visualization only
        in_bounds = (
                (final_grid[..., 0] >= -1) & (final_grid[..., 0] <= 1) &
                (final_grid[..., 1] >= -1) & (final_grid[..., 1] <= 1)
        ).float()

        # Sample patches
        patches = F.grid_sample(
            fm_expanded,
            final_grid,
            mode='nearest',       # exact nearest pixel
            padding_mode='zeros',
            align_corners=True    # consistent with our -1..1 corner definition
        )

        # Concatenate a mask channel
        in_bounds = in_bounds.unsqueeze(1)
        patches_with_mask = torch.cat([patches, in_bounds], dim=1)

        return patches_with_mask


if __name__ == "__main__":
    batch_size = 12
    num_channels = 4
    map_height, map_width = 24, 24

    # Random test image
    x = torch.rand((batch_size, num_channels, map_height, map_width))

    # We'll pick some (x_cord, y_cord) in [0..1] or [0..2] just for demonstration
    coords0 = torch.randint(0, 24, (batch_size, 16))  # or [0..2], etc.
    coords1 = torch.randint(0, 24, (batch_size, 16))
    x_cord = coords0
    y_cord = coords1

    units_masks = {'x_cord': x_cord, 'y_cord': y_cord}

    # Extract patches
    extractor = PatchExtractor(patch_size=15, patches_per_batch=16)
    patches = extractor.slices(x, units_masks)
    print("Patches shape:", patches.shape)  # => [B*M, C+1, 15, 15]

    # Test: check the center matches x[i,0,pos_x,pos_y]
    for i in range(batch_size):
        for j in range(16):
            pos_x = x_cord[i, j].item()  # int
            pos_y = y_cord[i, j].item()
            patch_idx = i * 16 + j

            # Check the center pixel (7,7) in a 15×15 patch
            # This should be exactly x[i, 0, pos_x, pos_y]
            for c in range(num_channels):
                patch_val = patches[patch_idx, c, 7, 7]
                original_val = x[i, c, pos_x, pos_y]
                assert patch_val == original_val, (
                    f"Mismatch at batch={i}, patch={j}, "
                    f"center=({pos_x},{pos_y}), "
                    f"patch_val={patch_val}, original_val={original_val}"
                )

    print("Center checks passed for all patches!")

    # If you also want to check offsets +/- 7 around (pos_x,pos_y),
    # you can do something like:
    for i in range(batch_size):
        for j in range(16):
            pos_x = x_cord[i, j].item()
            pos_y = y_cord[i, j].item()
            patch_idx = i * 16 + j
            # The patch's "center" is at (7,7)
            # We'll check all offsets in [-7..+7] so we don't go out of patch bounds
            for dx in range(-7, 8):
                for dy in range(-7, 8):
                    px = pos_x + dx
                    py = pos_y + dy
                    k = 7 + dx  # patch x index
                    l = 7 + dy  # patch y index
                    if (0 <= px < map_height) and (0 <= py < map_width):
                        for c in range(num_channels):
                            patch_val = patches[patch_idx, c, k, l]
                            original_val = x[i, c, px, py]
                            assert patch_val == original_val, (
                                f"Mismatch: patch={patch_idx}, offset=({dx},{dy}), "
                                f"patch_val={patch_val}, orig_val={original_val}"
                            )

    print("All offsets in every patch matched correctly!")
