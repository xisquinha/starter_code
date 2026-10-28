import random
import cv2 as cv
import numpy as np
import tp1

# -----------------
#   TASK 5
# -----------------

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
        num = int((end - start) * 0.1)
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

def precompute_shot_keyframes_with_sift(shots, video_capture, strategy="fst_middle_last", num_kfs=3):
    """
    Guarda em cache os keyframes, respetivos histogramas (cor) e descritores SIFT.
    """
    sift = cv.SIFT_create()
    cache = []
    
    for shot in shots:
        kf_indices = extract_keyframes(shot, strategy=strategy, num_kfs=num_kfs)
        
        shot_kfs_data = []
        for idx in kf_indices:
            video_capture.set(cv.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video_capture.read()
            
            if ret:
                # histograma de cor
                hist = tp1.bgr_histogram(frame)
                
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
    chi_dist = tp1.histogram_distance(kfA["hist"], kfB["hist"])
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


def compute_global_shot_similarity_matrix_full(shots, video_capture, kf_strategy="fst_middle_last", combination_method="max_mutual_matching", alpha=0.5):
    """
    Constrói a matriz global shot x shot com pré-computação total (Cor + SIFT em cache),
    aproveitando a simetria da matriz para maior velocidade.
    """
    cache = precompute_shot_keyframes_with_sift(shots, video_capture, strategy=kf_strategy)
    
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