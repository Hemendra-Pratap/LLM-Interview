from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import KMeans

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# 1. EMBEDDING FUNCTION
# ---------------------------
def get_embeddings(texts):
    """
    Convert list of texts into embeddings
    """
    return model.encode(texts, convert_to_tensor=True)


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
        score = util.cos_sim(ideal_emb[i], candidate_emb[i]).item()
        scores.append(round(score, 4))

    return scores


# ---------------------------
# 3. CLUSTERING FUNCTION
# ---------------------------
def cluster_scores(scores):
    """
    Cluster similarity scores into Strong / Average / Weak
    using KMeans (unsupervised learning)
    """
    if len(scores) < 3:
        # fallback if not enough data
        return ["Average"] * len(scores)

    X = np.array(scores).reshape(-1, 1)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    # sort clusters by center values
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
def evaluate_all_answers(ideal_answers, candidate_answers):
    """
    Full pipeline:
    - compute similarity
    - apply clustering
    - return structured results
    """

    if len(ideal_answers) != len(candidate_answers):
        raise ValueError("Ideal and candidate answers must have same length")

    # Step 1: similarity scores
    scores = compute_similarities(ideal_answers, candidate_answers)

    # Step 2: clustering
    categories = cluster_scores(scores)

    # Step 3: build output
    results = []
    for i in range(len(scores)):
        results.append({
            "score": scores[i],
            "percentage": round(scores[i] * 100, 1),
            "category": categories[i]
        })

    return results


# ---------------------------
# TEST RUN
# ---------------------------
if __name__ == "__main__":
    ideal_answers = [
        "An index improves query performance by avoiding full table scans.",
        "Normalization reduces redundancy and improves data integrity.",
        "A REST API allows communication between client and server using HTTP methods."
    ]

    candidate_answers = [
        "Indexing makes queries faster by using a lookup structure.",
        "Normalization organizes data into tables to remove duplication.",
        "API is used to connect frontend and backend."
    ]

    output = evaluate_all_answers(ideal_answers, candidate_answers)

    for i, res in enumerate(output):
        print(f"Q{i+1}: {res}")