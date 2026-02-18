from sec10q_rag.config import Settings
from sec10q_rag.rag import Sec10QRAG

if __name__ == "__main__":
    rag = Sec10QRAG(
        Settings(
            user_agent="Your Name (your.email@example.com)"
        )
    )

    answer, sources = rag.answer(
        ticker="AAPL",
        year=2024,
        quarter="Q1",
        question="What are the main risk factors mentioned?",
        k=5,
    )

    print(answer)
    print("\nSource metas:")
    for s in sources:
        print(s.meta)
