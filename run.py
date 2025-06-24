import json
import csv
import argparse
from src.hipporag import HippoRAG

def main(dataset_name: str):
    docs = []
    corpus_path = f"/ukp-storage-1/rolka1/thesis/data/{dataset_name}/corpus_sentence_256.jsonl"
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj["contents"])

    hipporag = HippoRAG(
        save_dir=f"/ukp-storage-1/rolka1/thesis/data/{dataset_name}/indexes/hipporag",
        llm_model_name='/storage/ukp/shared/shared_model_weights/models--Qwen3-8B',
        embedding_model_name='Qwen/Qwen3-Embedding-8B',
        llm_base_url="http://localhost:8000/v1"
    )

    # Run indexing
    hipporag.index(docs=docs)

    queries = []
    gold_answers = []
    gold_docs = []
    dev_path = f"/ukp-storage-1/rolka1/thesis/data/{dataset_name}/dev.jsonl"
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj["question"])
            gold_answers.append(obj.get("golden_answers", []))
            # metadata.context.sentences is List[List[str]], one sub-list per supporting doc
            ctx_lists = obj["metadata"]["context"]["sentences"]
            # join each sub-list of sentences into one string
            gold_docs.append([" ".join(sent_list) for sent_list in ctx_lists])

    solutions, response_messages, metadata, retrieval_metrics, qa_metrics = hipporag.rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers
    )

    results = []
    for idx, sol in enumerate(solutions):
        results.append({
            "id":               idx,
            "question":         sol.question,
            "gold_answer":      gold_answers[idx][0] if gold_answers[idx] else "",
            "predicted_answer": sol.answer,
        })

    out_csv = "results.csv"
    print(f"Writing results to {out_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "gold_answer", "predicted_answer"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    args = parser.parse_args()
    main(args.dataset_name)