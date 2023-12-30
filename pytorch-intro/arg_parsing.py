import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='The is a demo')
    # Here we add an argument to the parser, specifying the expected type, 
    # a help message, etc.
    parser.add_argument(
        '-batch_size', 
        type=int, 
        required=True, 
        help='Please provide -batch_size'
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    print(f'The provided batch size is: {args.batch_size}')

    return 

if __name__ == '__main__':
    main()