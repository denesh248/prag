from src.rag_pipeline import RAGPipeline


def main():

    rag = RAGPipeline()

    print("\nRAG system is ready!")

    while True:

        query = input("\nAsk a question: ")

        if query.lower() in [
            "exit",
            "quit"
        ]:
            break

        result = rag.ask(
            query
        )

        print("\nAnswer:")
        print(
            result["answer"]
        )

        print("\nSources:")

        for source in result["sources"]:

            print(
                f"- {source['source']} "
                f"(page {source['page']}) "
                f"score={source['score']}"
            )


if __name__ == "__main__":
    main()