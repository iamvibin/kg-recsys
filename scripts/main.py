import argparse
from preprocess import preprocess
from sim_interactions_from_kg import generate_user_item_pair
from generate_blocking_target import generate_blocking_target

def start_datageneration(args):
    preprocess(args)
    generate_user_item_pair(args)
    generate_blocking_target(args)

    print("data generation complete")


