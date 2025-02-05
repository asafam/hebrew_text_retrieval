from translation.api.translate_batch import retrieve_batch_results
import argparse


def main():
    parser = argparse.ArgumentParser(description="Check the status of a batch translation job.")
    retrieve_batch_results()

