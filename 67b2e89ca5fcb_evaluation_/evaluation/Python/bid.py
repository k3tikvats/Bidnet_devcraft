#!/usr/bin/python
# -*- coding:utf8 -*-
import random
import joblib
from dataloader import Data_processor
from bidRequest import BidRequest
from time import time
from dataloader import XGBoost
import json 

class Bid(object):

    def __init__(self):

        self.model_bid = XGBoost("xgb_bid_model.joblib")
        self.model_price = XGBoost("xgb_price_model.joblib")


    def get_bid_price(self, bidRequest):
        X=Data_processor(bidRequest=bidRequest)
        # t=time()
        y_bid_pred = self.model_bid.predict([X])
        if y_bid_pred:
            y_price=self.model_price.predict([X])
        else:
            y_price=y_bid_pred
        # print(time()-t)
        return y_price

if __name__=='__main__':

    # for sample testing
    b=BidRequest()
    b.timestamp='20150314101523123'
    b.ip_address='192.143.16.*'
    b.region='94'
    b.city='9'
    b.ad_slot_id = None
    b.ad_slot_width = '30'
    b.ad_slot_height = '15'
    b.ad_slot_visibility= '2'
    b.ad_slot_format = '2'
    b.ad_slot_floor_price = '5'
    b.advertiser_id ='1458'
    b.user_tags='13042,10110'
    ob=Bid()
    t=time()
    ob.get_bid_price(b)
    print('time taken',time()-t)
    

