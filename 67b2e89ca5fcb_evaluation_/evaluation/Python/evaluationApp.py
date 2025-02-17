#!/usr/bin/python
# -*- coding:utf8 -*-
import csv

import bidRequest

from bid import Bid

import argparse


class TestData:
    def __init__(self):
        self.bid_request = None
        self.bid_price = 0
        self.paying_price = 0
        self.clicks = 0
        self.conversions = 0
        self.weight = 0


class BudgetInfo:
    def __init__(self, budget, log_lines):
        self.budget = budget
        self.log_lines = log_lines


class TestDataParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'r')
        self.reader = csv.reader(self.file, delimiter='\t')

    def read_next_test_data(self):
        try:
            line = next(self.reader)
            bid_request = bidRequest.BidRequest()
            bid_request.bid_id = line[0]
            bid_request.timestamp = line[1]
            bid_request.visitor_id = line[3]
            bid_request.user_agent = line[4]
            bid_request.ip_address = line[5]
            bid_request.region = line[6]
            bid_request.city = line[7]
            bid_request.ad_exchange = line[8]
            bid_request.domain = line[9]
            bid_request.url = line[10]
            bid_request.anonymous_url_id = line[11]
            bid_request.ad_slot_id = line[12]
            bid_request.ad_slot_width = line[13]
            bid_request.ad_slot_height = line[14]
            bid_request.ad_slot_visibility = line[15]
            bid_request.ad_slot_format = line[16]
            bid_request.ad_slot_floor_price = line[17]
            bid_request.creative_id = line[18]
            bid_request.advertiser_id = line[22]
            bid_request.user_tags = line[23]

            test_data = TestData()
            test_data.bid_request = bid_request
            test_data.bid_price = int(line[19])
            test_data.paying_price = int(line[20])
            test_data.clicks = int(line[24])
            test_data.conversions = int(line[25])
            test_data.weight = self.get_weight(bid_request.advertiser_id)

            return test_data
        except StopIteration:
            return None

    @staticmethod
    def get_weight(advertiser_id):
        weights = {
            "1458": 0,
            "3358": 2,
            "3386": 0,
            "3427": 0,
            "3476": 10
        }
        return weights.get(advertiser_id, 0)

    def close(self):
        self.file.close()


def initialize_budgets1():
    return {
        "1458": BudgetInfo(15000.0, 0),
        "3358": BudgetInfo(16000.0, 0),
        "3386": BudgetInfo(15500.0, 0),
        "3427": BudgetInfo(16500.0, 0),
        "3476": BudgetInfo(14500.0, 0)
    }

def initialize_budgets2():
    return {
        "1458": BudgetInfo(5000.0, 0),
        "3358": BudgetInfo(4800.0, 0),
        "3386": BudgetInfo(3900.0, 0),
        "3427": BudgetInfo(4500.0, 0),
        "3476": BudgetInfo(4200.0, 0)
    }

def initialize_budgets3():
    return {
        "1458": BudgetInfo(20000.0, 0),
        "3358": BudgetInfo(2000.0, 0),
        "3386": BudgetInfo(15000.0, 0),
        "3427": BudgetInfo(1800.0, 0),
        "3476": BudgetInfo(8000.0, 0)
    }

def __evaluate(args):
    print("Starting the bidding process...")
    try:
        print(f"Evaluating for Budget Set: {args.budget_set}")
        parser = TestDataParser('/Users/anuragpandey/Downloads/evaluation/test.dataset.10L.txt')
        bid = Bid()
        if(args.budget_set == 1):
            budgets = initialize_budgets1()
        elif(args.budget_set == 2):
            budgets = initialize_budgets2()
        elif(args.budget_set == 3):
            budgets = initialize_budgets3()  
        else:
            print("Invalid budget set!")
            raise ValueError("Invalid budget set!") 
        exhausted_budgets = {}
        score = 0
        log_lines = 0

        while len(exhausted_budgets) != len(budgets) and (test_data := parser.read_next_test_data()) is not None:
            log_lines += 1
            advertiser_id = test_data.bid_request.advertiser_id
            budget_info = budgets[advertiser_id]
            if budget_info.budget <= 0:
                continue

            import time
            start_time = time.time()
            bid_price = bid.get_bid_price(test_data.bid_request)
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            if duration > 5:
                print(f"WARNING: get_bid_price took {duration:.2f} ms")

            if bid_price > test_data.paying_price:
                score += test_data.clicks + test_data.weight * test_data.conversions
                budget_info.budget -= test_data.paying_price / 1000.0
            budget_info.log_lines = log_lines

            if budget_info.budget <= 0:
                exhausted_budgets[advertiser_id] = budget_info.budget

        print(f"Score: {score:,}, Evaluated log lines: {log_lines:,}")
        print("Remaining Budgets:")
        for advertiser_id, budget_info in budgets.items():
            print(
                f"Advertiser {advertiser_id}: {budget_info.budget:.3f}, Evaluated log lines: {budget_info.log_lines:,}")

        parser.close()
        print("Bidding process completed successfully.")
    except FileNotFoundError:
        print("Failed to create TestDataParser")
    except ValueError:
        print("Failed to parse integer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budget Set Input")
    parser.add_argument("--budget_set", type=int, default=1)
    args = parser.parse_args()
    __evaluate(args)


        
