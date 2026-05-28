import argparse
from translation.api.translate_batch_gemini import retrieve_batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve completed Gemini batch translation results."
    )
    parser.parse_args()
    retrieve_batch_results()


if __name__ == "__main__":
    main()
