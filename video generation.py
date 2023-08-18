# Game of life video generation
import numpy as np
import cv2
from tqdm import tqdm


def count_neighbors(grid, y, x):
    res = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= y + i < len(grid) and 0 <= x + j < len(grid[0]):
                res += grid[y + i][x + j]
    return res - grid[y][x]


def next_gen(grid):
    res = [[0 for __ in grid[0]] for _ in grid]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                if count_neighbors(grid, i, j) in [2, 3]:
                    res[i][j] = 1
            else:
                if count_neighbors(grid, i, j) == 3:
                    res[i][j] = 1
    return res


def generate_frames(num_frames):
    grid = np.random.randint(2, size=(256, 256))
    frames = []
    for _ in tqdm(range(num_frames)):
        grid = next_gen(grid)
        frame = np.array([[[i, i, i] for i in j] for j in grid], dtype=np.uint8) * 255
        frames.append(frame)
    return frames


def create_video_from_bitmaps(frames, output_file='output.mov'):
    # Assuming all frames have the same shape.
    height, width, _ = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'png ')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

    for frame in frames:
        out.write(frame)  # Convert grayscale to BGR for VideoWriter

    out.release()


if __name__ == '__main__':
    frames = generate_frames(10000)  # Example 100 frames
    create_video_from_bitmaps(frames)
