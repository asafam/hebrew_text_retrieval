from translation.api.translate_batch import check_batch_status
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Check the status of a batch translation job.")

    parser.add_argument('--sleep_time', type=int, default=60, help="Time to sleep between status checks.")
    
    args = parser.parse_args()

    # Wait & Check Status
    while True:
        status = check_batch_status()
        if status in ["completed", "failed"]:
            break
        time.sleep(args.sleep_time)  # Poll every x seconds