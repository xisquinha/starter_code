###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## Student 1 (number)
## Student 2 (number)
##
###############################################################################

import glob
import json
import os

import cv2 as cv
import numpy as np

import grouping   # group_shots: union-find grouping of shots (Part 3, Task 7)
import metrics

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")


def read_frames(path, size=(320, 180)):
    """Read every frame of a video, resized to `size`. Returns a list of BGR images."""
    cap = cv.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv.resize(frame, size, interpolation=cv.INTER_AREA))
    cap.release()
    return frames


def merge_close(pred, min_gap=5):
    merged = []
    for f in pred:
        if not merged or f - merged[-1] > min_gap:
            merged.append(f)
    return merged

def detect_cuts(signal, T, min_gap=5):
    return merge_close([i + 1 for i, v in enumerate(signal) if v > T], min_gap)


def process_video(frames, gt_shots):
    """Return (transitions, scenes) for one video.

    transitions: list of {"type": "cut", "start": t, "end": t}, t = first frame of the new shot (Parts 1 and 2)
    scenes:      dict {str(shot id): label} for the ground-truth shots (Part 3)
    """
    transitions, scenes = [], {}
    # TODO: Parts 1, 2 and 3. Everything else (writing the output, evaluation, summary table) is done for you.
    
    T_hist = 0.122

    d_hist = []
    prev_frame = frames[0]
    for i in range(len(frames) - 1):

        frame = frames[i]

        hist1 = bgr_histogram(prev_frame)
        hist2 = bgr_histogram(frame)
    
        pixel_dif = pixel_difference(prev_frame, frame)
        hist_dif = histogram_distance(hist1, hist2)
        ecr = edge_change_ration(prev_frame, frame)
    
        d_hist.append(hist_dif)
        prev_frame = frame

    cuts = detect_cuts(d_hist, T_hist)

    for cut in cuts:
        transitions.append({"type": "cut", "start":cut, "end": cut})
    
    return transitions, scenes


## assuming that f1 and f2 are in BGR
## mean absolute difference of grey levels. This is your baseline.
def pixel_difference(f1, f2):
    f1_gray = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    f2_gray = cv.cvtColor(f2, cv.COLOR_BGR2GRAY)

    dif = abs(f1_gray - f2_gray)

    return np.mean(dif)


def bgr_histogram(img, bins=256):
    hists = []
    for c in range(3):
        h = cv.calcHist([img], [c], None, [bins], [0, 256]).flatten()
        hists.append(h)
    h = np.concatenate(hists)
    return h / h.sum()  # normalize so image size doesn't matter

# Function to calculate Chi-distance    
def histogram_distance(h1, h2):
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
   
    mask = (h1 + h2) > 0
    chi = 0.5 * np.sum([((xi - yi) ** 2) / (xi + yi) 
                        for (xi, yi) in zip(h1[mask], h2[mask])])

    return chi



def edge_change_ration(f1, f2):
    f1_gray = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    f2_gray = cv.cvtColor(f2, cv.COLOR_BGR2GRAY)

    canny1 = cv.Canny(f1_gray, 40, 200)
    canny2 = cv.Canny(f2_gray, 40, 200)

    kernel = np.ones((5, 5), np.uint8)
    f1_dilate = cv.dilate(canny1, kernel, iterations=1)
    f2_dilate = cv.dilate(canny2, kernel, iterations=1)

    x_in = np.sum(np.array(canny2) & ~np.array(f1_dilate))
    x_out = np.sum(np.array(canny1) & ~np.array(f2_dilate))

    a = 0
    b = 0

    ## check if any of the frames don't have any edge
    if(np.sum(canny1) != 0):
        a = x_out/np.sum(canny1)
    elif (np.sum(canny2) != 0):
        b = x_in/np.sum(canny2)
    
    ecr = max(a, b)

    return ecr


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}                                    # video name -> counts from metrics.evaluate_video
    for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video_path))[0]
        gt = json.load(open(os.path.join(INPUT_DIR, name + ".json")))

        frames = read_frames(video_path)
        
        transitions, scenes = process_video(frames, gt["shots"])
        pred = {"video": name + ".mp4", "transitions": transitions, "scenes": scenes}
        with open(os.path.join(OUTPUT_DIR, name + ".json"), "w") as f:
            json.dump(pred, f, indent=1, default=int)    # default=int: accepts NumPy integers
        results[name] = metrics.evaluate_video(gt, pred)

    # summary table (Section 8): one row per video, then the dev / test totals
    dev = [results[n] for n in sorted(results) if n.startswith("dev")]
    test = [results[n] for n in sorted(results) if n.startswith("test")]
    metrics.print_table([(n, results[n]) for n in sorted(results)]
                        + [("DEV", metrics.sum_counts(dev)), ("TEST", metrics.sum_counts(test))])
