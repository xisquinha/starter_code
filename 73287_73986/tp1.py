###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## Beatriz Guerreiro (73287)
## Francisca Inácio (73986)
##
###############################################################################

import glob
import json
import os
import random
import math
import matplotlib.pyplot as plt 

import cv2 as cv
import numpy as np

import grouping   # group_shots: union-find grouping of shots (Part 3, Task 7)
import metrics

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")
GAP = 4
T_HIST = 0.122
T_SIFT = 0.11
T_SCENES = 0.8 


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
    """
    Merge cuts that are too close to each other, keeping only the first one.

    This helps filtering the cuts, since the change between shots can happen in
    more than one frame, so the algorithm would detect the same cut more than
    once. A cut is dropped if it is at most 'min_gap' frames after the last cut.

    Args:
        pred (list): list of cut frame indices
        min_gap (int): maximum distance at which two cuts are considered to be
        the same one

    Returns:
        list: the same list of cut indices, but with very close cuts merged
    """
    merged = []
    for f in pred:
        if len(merged) == 0 or f - merged[-1] > min_gap:
            merged.append(f)
    return merged


def detect_cuts(signal, T=T_HIST, min_gap=5):
    """
    Detect in which frames a cut occurs.

    Args:
        signal (list): dissimiarity values between consecutives frames
        T (float): threshold for the same dissimilarity as the signal. A cut
        is detected when the value is higher than T.
        min_gap (int): the gap to pass to the merge_close function

    Returns:
        list: indices of the cuts, each index is the first frame of the new shot
    """
    return merge_close([i + 1 for i, v in enumerate(signal) if v > T], min_gap)


def sift_matches(frame1, frame2):
    """
    Compare the local features of two images using SIFT and Lowe's ratio test.
    If either image has fewer than 2 descriptors, four empty lists are returned.

    Args:
        frame1 (numpy.ndarray): first image to compare (in BGR)
        frame2 (numpy.ndarray): second image to compare (in BGR)

    Returns:
        list: matches that passed the Lowe's ratio test
        tuple: all candidate matches
        tuple: detected keypoints in the frame1
        tuple: detected keypoints in the frame2
    """
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return [], [], [], []
    
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return good, matches, kp1, kp2



def verification_false_alarms(f1, f2, T=T_SIFT):
    """
    Check whether a detected cut is a false alarm using SIFT.
    f1 and f2 are frames taken before and after the candidate cut. If they look
    very alike, the cut is probably a false alarm.

    Args:
        f1 (numpy.ndarray): first frame, in BGR
        f2 (numpy.ndarray): second frame, in BGR
        T (float): minimum ratio of good matches / candidate matches, for both
         frames to be considered similar (Lowe's test)

    Returns:
        bool: true if the frames are similar, false otherwise
    """
    good, matches, _, _ = sift_matches(f1, f2)
    if len(matches) == 0:
        return False
    else:
        return len(good) / len(matches) > T


def process_video(frames, gt_shots, verify=True, use_gt=True):
    """
    Detect cuts and group shots into recurring scenes for one video.

    Cuts come from thresholding the chi distance between the color histograms 
    of consecutive frames, then filtered with SIFT. Shots are then grouped by 
    similarity.

    Args:
        frames (list): frames of the video
        gt_shots (list): ground-truth shots, dicts with "id", "start", "end"
        verify (bool): if true run the video frames through SIFT
        use_gt (bool): if true, group the ground-truth shots, otherwise use the
            shots built from the detected cuts
    
    Returns:
        list: transitions, dicts {"type": "cut", "start": t, "end": t}, t = first 
        frame of the new shot (Parts 1 and 2)
        dict: {str(shot id): scene label} for the ground-truth shots (Part 3)
    """
    transitions, scenes = [], {}
    
    d_hist = []
    for i in range(len(frames) - 1):

        hist1 = bgr_histogram(frames[i])
        hist2 = bgr_histogram(frames[i+1])
    
        hist_dif = histogram_distance(hist1, hist2)
    
        d_hist.append(hist_dif)

    cuts = detect_cuts(d_hist, T_HIST)

    for cut in cuts:

        prev_cut = 0
        pos_cut = len(frames) - 1
        if(cut > GAP-1): prev_cut = cut - GAP
        if(cut + GAP < len(frames) - 1): pos_cut = cut + GAP
        
        if not verify or not verification_false_alarms(frames[prev_cut], frames[pos_cut]):
            transitions.append({"type": "cut", "start":cut, "end": cut})

    if use_gt: shots = gt_shots
    else: shots = transitions_to_shots(transitions, len(frames))
      # Recurring scenes detection - part 3
    shot_similarities = compute_global_shot_similarity_matrix_full(
        shots, 
        frames, 
        kf_strategy="single_middle",
        combination_method="max_mutual_matching",
        alpha=0.5
    )

    labels = grouping.group_shots(shot_similarities, threshold=T_SCENES)
    scenes = {str(s["id"]): label for s, label in zip(shots, labels)}
    
    return transitions, scenes


## assuming that f1 and f2 are in BGR
## mean absolute difference of grey levels. This is your baseline.
def pixel_difference(f1, f2):
    """
    Mean absolute difference of gray levels between two BGR frames

    Args:
        f1 (numpy.ndarray): first frame, in BGR
        f2 (numpy.ndarray): second frame, in BGR

    Returns:
        float: result of the difference [0, 255]
    """
    f1_gray = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    f2_gray = cv.cvtColor(f2, cv.COLOR_BGR2GRAY)

    dif = np.abs(f1_gray.astype(np.int16) - f2_gray.astype(np.int16))

    return np.mean(dif)


def bgr_histogram(img, bins=256, show_hist=False):
    """
    Generates a histogram color for a BGR image

    Args:
        img (numpy.ndarray): BGR image
        bins (int): bins per channel
        show_hist (bool): if true, return histograms as a list of 3 arrays for 
        plotting

    Returns:
        numpy.ndarray: vector of 3*bins values normalised, so the image size 
        does not matter, or a list of 3 arrays if show_hist
    """
    
    hists = []
    for c in range(3):
        h = cv.calcHist([img], [c], None, [bins], [0, 256]).flatten()
        hists.append(h)

    if show_hist:
        return hists

    h = np.concatenate(hists)
    return h / h.sum()  # normalize so image size doesn't matter

# Function to calculate Chi-distance    
def histogram_distance(h1, h2):
    """
    Chi-square distance between two histograms.
    
    Args:
        h1 (numpy.ndarray): histogram of first image
        h2 (numpy.ndarray): histogram of second image

    Returns:
        float: chi distance
    """

    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
   
    mask = (h1 + h2) > 0
    chi = 0.5 * np.sum((h1[mask] - h2[mask]) ** 2 / (h1[mask] + h2[mask]))

    return chi



def edge_change_ration(f1, f2):
    """
    Edge Change Ratio between two BGR frames.
    Canny edges are extracted from both frames (thresholds 40/200). Edge
    pixels of f2 not within a 5x5 dilation of f1's edges are "entering",
    and edge pixels of f1 not near f2's edges are "exiting".

    Args:
        f1 (numpy.ndarray): first frame, in BGR
        f2 (numpy.ndarray): second frame, in BGR

    Returns:
        float: max(exiting / edges in f1, entering / edges in f2)
    """

    f1_gray = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    f2_gray = cv.cvtColor(f2, cv.COLOR_BGR2GRAY)

    canny1 = cv.Canny(f1_gray, 40, 200)
    canny2 = cv.Canny(f2_gray, 40, 200)

    kernel = np.ones((5, 5), np.uint8)
    f1_dilate = cv.dilate(canny1, kernel, iterations=1)
    f2_dilate = cv.dilate(canny2, kernel, iterations=1)

    x_in = np.sum(np.array(canny2) & ~np.array(f1_dilate))
    x_out = np.sum(np.array(canny1) & ~np.array(f2_dilate))

    n1 = np.sum(canny1)
    n2 = np.sum(canny2)

    a = x_out / n1 if n1 > 0 else 0.0
    b = x_in / n2 if n2 > 0 else 0.0
    
    ecr = max(a, b)

    return ecr


## aux functions
def print_metrics(cuts1, cuts2):
    """
    Print TP, FP, FN, precision, recall and F1 for two lists of cut indices.
    """

    tp, fp, fn = metrics.match_cuts(cuts2, cuts1)
    precision, recall, f1 = metrics.precision_recall_f1(tp, fp, fn)

    print("true positives:", tp)
    print("false positives:", fp)
    print("missed detections:", fn)

    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)


def cut_list_to_transitions(list):
    """
    Convert a list of cut indices into the output format of the project.

    Returns:
        dict: {"transitions": [{"type": "cut", "start": n, "end": n}, "scenes": {}}
    """
     
    cuts = []
    for cut in list:
        cuts.append({"type": "cut", "start": cut, "end": cut})

    return {"transitions":cuts, "scenes":{}}


def transitions_to_shots(trans, num_frames):
    """
    Build shots from the detected cuts.
    Each shot goes from one cut to the frame before the next. 

    Returns:
        list: dicts {"id", "start", "end", "scene"}
    """

    trans = [t["start"] for t in trans]
    bounds = [0] + trans + [num_frames]
    shots = []

    for i in range(len(bounds) - 1):
        shots.append({"id":i, "start":bounds[i], "end":trans[i+1]-1, "scene":"S"+str(i)})
    return shots


# diferentes estratégias para escolher keyframes de um shot
def extract_keyframes(shot, strategy="single", num_kfs=3):
    start = shot["start"]
    end = shot["end"]
    keyframes = []
    
    if strategy == "single_fst":  # Apenas o 1º frame
        keyframes.append(start)

    elif strategy == "single_middle":  # Apenas o frame do meio
        mid = (start + end) // 2
        keyframes.append(mid)

    elif strategy == "fst_middle_last":  # Usa o frame do meio, o primeiro e o último
        mid = (start + end) // 2
        keyframes.append(start)
        keyframes.append(mid)
        keyframes.append(end)
        
    elif strategy == "multiple_eq":  # Vários keyframes espaçados igualmente
        num = min(num_kfs, end - start)
        keyframes = [int(x) for x in np.linspace(start, end, num)]

    elif strategy == "random":  # n aleatórios
        num = min(num_kfs, end - start)
        for i in range(num):
            keyframes.append(random.randint(start, end))

    elif strategy == "random_10%":  # 10% aleatórios do total de frames no shot 
        num = int((end - start + 1) * 0.1)
        for i in range(num):
            keyframes.append(random.randint(start, end))
        
    return keyframes


# diferentes metodos para combinar as similarities num só valor
def combine_keyframe_similarities(similarities_matrix, method="best_pair"):
    matrix = np.array(similarities_matrix)
    
    if method == "best_pair":
        return np.max(matrix)
        
    elif method == "average":
        return np.mean(matrix)

    elif method == "sum":
        return np.sum(matrix)
        
    elif method == "threshold_count" or method == "vote":
        # Contar quantos pares passam de um limiar de similaridade (ex: > 0.6)
        threshold = 0.6
        matching_pairs = np.sum(matrix > threshold)
        return float(matching_pairs)
        
    elif method == "weighted_threshold_count":
        score = 0.0
        for val in matrix.flatten():
            if val > 0.8:
                score += 1.0   # strong
            elif val > 0.6:
                score += 0.5   # medium
            elif val > 0.4:
                score += 0.2   # weak
            else:
                score += 0.0
        
        score = score / matrix.size if matrix.size > 0 else 0.0
        return score

    elif method == "tiered_weighted_average":
        total_score = 0.0
        total_weight = 0.0

        for val in matrix.flatten():
            if val > 0.7:        # strong
                weight = 1.0
            elif val > 0.4:      # medium
                weight = 0.5
            else:                # weak
                weight = 0.1

            total_score += val * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    elif method == "max_mutual_matching":
        if matrix.size == 0:
            return 0.0

        # Melhor match de cada linha (Shot 1 -> Shot 2)
        best_from_shot1 = np.max(matrix, axis=1)
        # Melhor match de cada coluna (Shot 2 -> Shot 1)
        #best_from_shot2 = np.max(matrix, axis=0)

        #all_best_matches = np.concatenate([best_from_shot1, best_from_shot2])
        return float(np.mean(best_from_shot1))
        
    return 0.0


# -----------------
#   TASK 6
# -----------------

def precompute_shot_keyframes_with_sift(shots, frames, strategy="fst_middle_last", num_kfs=3):
    """
    Guarda em cache os keyframes, respetivos histogramas (cor) e descritores SIFT.
    """
    sift = cv.SIFT_create()
    cache = []
    
    for shot in shots:
        kf_indices = extract_keyframes(shot, strategy=strategy, num_kfs=num_kfs)
        
        shot_kfs_data = []
        for idx in kf_indices:

            frame = frames[idx]
            
            # histograma de cor
            hist = bgr_histogram(frame)
            
            # SIFT
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            
            shot_kfs_data.append({
                "index": idx,
                "image": frame,
                "hist": hist,
                "kp": kp,
                "des": des
            })
                
        cache.append(shot_kfs_data)
        
    return cache


def cached_sift_similarity(des1, des2, ratio=0.7, max_expected_matches=50):
    """
    Calcula a similaridade SIFT usando descritores já pré-computados e FLANN.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0
        
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    raw_matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good_matches.append(pair[0])
            
    num_good = len(good_matches)
    return float(min(1.0, num_good / max_expected_matches))


def cached_keyframe_similarity(kfA, kfB, alpha=0.5):
    """
    Calcula a similaridade total (Cor + SIFT) entre dois keyframes usando a cache.
    alpha: peso da cor (0 a 1). O SIFT fica com (1 - alpha).
    """
    chi_dist = histogram_distance(kfA["hist"], kfB["hist"])
    sim_col = max(0.0, 1.0 - chi_dist)
    
    sim_sift = cached_sift_similarity(kfA["des"], kfB["des"])
    
    total_sim = (alpha * sim_col) + ((1 - alpha) * sim_sift)
    
    return float(total_sim)


def compute_shot_similarity_from_cache(cacheA, cacheB, combination_method="max_mutual_matching", alpha=0.5):
    """
    Calcula a similaridade final entre dois shots usando a cache completa (Cor + SIFT).
    """
    if not cacheA or not cacheB:
        return 0.0
        
    kf_matrix = []
    for kfA in cacheA:
        row = []
        for kfB in cacheB:
            sim = cached_keyframe_similarity(kfA, kfB, alpha=alpha)
            row.append(sim)
        kf_matrix.append(row)
        
    return float(combine_keyframe_similarities(kf_matrix, method=combination_method))


def compute_global_shot_similarity_matrix_full(shots, frames, kf_strategy="fst_middle_last", combination_method="max_mutual_matching", alpha=0.5):
    """
    Constrói a matriz global shot x shot com pré-computação total (Cor + SIFT em cache),
    aproveitando a simetria da matriz para maior velocidade.
    """
    cache = precompute_shot_keyframes_with_sift(shots, frames, strategy=kf_strategy)
    
    num_shots = len(shots)
    global_matrix = np.zeros((num_shots, num_shots))
    
    for i in range(num_shots):
        for j in range(num_shots):
            if i < j:
                global_matrix[i, j] = compute_shot_similarity_from_cache(
                    cache[i], cache[j], combination_method=combination_method, alpha=alpha
                )
            elif j < i:
                global_matrix[i, j] = global_matrix[j, i]
            else: 
                global_matrix[i, j] = 1.0
            
    return global_matrix

def show_images(images, per_row=5, size=3, scenes=None, cmaps = []):
    n = len(images)
    rows = math.ceil(n / per_row)          # 37 / 5 -> 8 rows

    _, axes = plt.subplots(rows, per_row, figsize=(per_row * size, rows * size))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            if(len(cmaps) == 0): cmap = None
            else: cmap = cmaps[i]

            if cmap == None: img = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
            else: img = images[i]
            
            ax.imshow(img, cmap = cmap)
            if scenes is not None:
                ax.set_title("Shot "+str(i)+", labled:"+str(scenes[str(i)]), fontsize=9)
        ax.axis("off")                      # also hides the unused slots (3 empty ones here)

    plt.tight_layout()
    plt.show()


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
