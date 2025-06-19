import os
from typing import List
import json
import argparse
import logging
import csv
from src.hipporag import HippoRAG


def main():

    docs = []
    with open("../../data/hotpotqa/corpus_sentence_6.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj["contents"])

    save_dir = 'outputs/demo_llama'
    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name='Qwen/Qwen3-0.6B',
                        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                        llm_base_url="http://localhost:8000/v1")

    # Run indexing
    hipporag.index(docs=docs)

    queries = []
    gold_answers = []
    gold_docs = []
    with open("../../data/hotpotqa/train.jsonl", "r", encoding="utf-8") as f:
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

    # 2) collect our results
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
    main()
