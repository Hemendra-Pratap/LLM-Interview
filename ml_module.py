import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import util

# ============================
# 🔷 LAZY MODEL LOADING
# ============================

model = None

def get_model():
    global model

    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


# ---------------------------
# 1. EMBEDDING FUNCTION
# ---------------------------
def get_embeddings(texts):
    """
    Convert list of texts into embeddings
    """
    return get_model().encode(texts, convert_to_tensor=True)


# ---------------------------
# 2. SIMILARITY FUNCTION
# ---------------------------
def compute_similarities(ideal_answers, candidate_answers):
    """
    Compute cosine similarity for each (ideal, candidate) pair
    """

    ideal_emb = get_embeddings(ideal_answers)
    candidate_emb = get_embeddings(candidate_answers)

    scores = []

    for i in range(len(ideal_answers)):
        score = util.cos_sim(
            ideal_emb[i],
            candidate_emb[i]
        ).item()

        scores.append(round(score, 4))

    return scores


# ---------------------------
# 3. CLUSTERING FUNCTION
# ---------------------------
def cluster_scores(scores):
    """
    Cluster similarity scores into:
    Strong / Average / Weak
    """

    if len(scores) < 3:
        return ["Average"] * len(scores)

    X = np.array(scores).reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    # Sort cluster centers
    order = np.argsort(centers)

    label_map = {
        order[0]: "Weak",
        order[1]: "Average",
        order[2]: "Strong"
    }

    return [label_map[label] for label in labels]


# ---------------------------
# 4. MAIN EVALUATION FUNCTION
# ---------------------------
def evaluate_all_answers(
    ideal_answers,
    candidate_answers
):
    """
    Full evaluation pipeline
    """

    if len(ideal_answers) != len(candidate_answers):
        raise ValueError(
            "Ideal and candidate answers must have same length"
        )

    # Step 1: similarity
    scores = compute_similarities(
        ideal_answers,
        candidate_answers
    )

    # Step 2: clustering
    categories = cluster_scores(scores)

    # Step 3: build response
    results = []

    for i in range(len(scores)):
        results.append({
            "score": scores[i],
            "percentage": round(max(scores[i], 0) * 100, 1),
            "category": categories[i]
        })

    return results


# ---------------------------
# TEST RUN
# ---------------------------
if __name__ == "__main__":

    ideal_answers = [
        "An index improves query performance.",
        "Normalization reduces redundancy.",
        "REST API enables communication."
    ]

    candidate_answers = [
        "Indexing makes queries faster.",
        "Normalization removes duplication.",
        "API connects frontend and backend."
    ]

    output = evaluate_all_answers(
        ideal_answers,
        candidate_answers
    )

    for i, res in enumerate(output):
        print(f"Q{i+1}: {res}")
