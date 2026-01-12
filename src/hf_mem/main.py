import argparse
import asyncio
from hf_mem.hub.hf_client import run

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Hugging Face model memory usage.")
    parser.add_argument("--model-id", required=True, help="Model ID on the Hugging Face Hub")
    parser.add_argument("--revision", default="main", help="Model revision")
    parser.add_argument("--json-output", action="store_true", help="Output as JSON")
    parser.add_argument("--ignore-table-width", action="store_true", help="Ignore max table width")

    args = parser.parse_args()

    try:
        asyncio.run(
            run(
                model_id=args.model_id,
                revision=args.revision,
                json_output=args.json_output,
                ignore_table_width=args.ignore_table_width,
            )
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()