#!/usr/bin/python
# -*- coding:utf8 -*-

class BidRequest(object):
    def __init__(self):
        self.bid_id = ""
        self.timestamp = ""
        self.visitor_id = ""
        self.user_agent = ""
        self.ip_address = ""
        self.region = ""
        self.city = ""
        self.ad_exchange = ""
        self.domain = ""
        self.url = ""
        self.anonymous_url_id = ""
        self.ad_slot_id = ""
        self.ad_slot_width = ""
        self.ad_slot_height = ""
        self.ad_slot_visibility = ""
        self.ad_slot_format = ""
        self.ad_slot_floor_price = ""
        self.creative_id = ""
        self.advertiser_id = ""
        self.user_tags = ""

    def get_bid_id(self):
        return self.bid_id

    def set_bid_id(self, bid_id):
        self.bid_id = bid_id

    def get_timestamp(self):
        return self.timestamp

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def get_visitor_id(self):
        return self.visitor_id

    def set_visitor_id(self, visitor_id):
        self.visitor_id = visitor_id

    def get_user_agent(self):
        return self.user_agent

    def set_user_agent(self, user_agent):
        self.user_agent = user_agent

    def get_ip_address(self):
        return self.ip_address

    def set_ip_address(self, ip_address):
        self.ip_address = ip_address

    def get_region(self):
        return self.region

    def set_region(self, region):
        self.region = region

    def get_city(self):
        return self.city

    def set_city(self, city):
        self.city = city

    def get_ad_exchange(self):
        return self.ad_exchange

    def set_ad_exchange(self, ad_exchange):
        self.ad_exchange = ad_exchange

    def get_domain(self):
        return self.domain

    def set_domain(self, domain):
        self.domain = domain

    def get_url(self):
        return self.url

    def set_url(self, url):
        self.url = url

    def get_anonymous_url_id(self):
        return self.anonymous_url_id

    def set_anonymous_url_id(self, anonymous_url_id):
        self.anonymous_url_id = anonymous_url_id

    def get_ad_slot_id(self):
        return self.ad_slot_id

    def set_ad_slot_id(self, ad_slot_id):
        self.ad_slot_id = ad_slot_id

    def get_ad_slot_width(self):
        return self.ad_slot_width

    def set_ad_slot_width(self, ad_slot_width):
        self.ad_slot_width = ad_slot_width

    def get_ad_slot_height(self):
        return self.ad_slot_height

    def set_ad_slot_height(self, ad_slot_height):
        self.ad_slot_height = ad_slot_height

    def get_ad_slot_visibility(self):
        return self.ad_slot_visibility

    def set_ad_slot_visibility(self, ad_slot_visibility):
        self.ad_slot_visibility = ad_slot_visibility

    def get_ad_slot_format(self):
        return self.ad_slot_format

    def set_ad_slot_format(self, ad_slot_format):
        self.ad_slot_format = ad_slot_format

    def get_ad_slot_floor_price(self):
        return self.ad_slot_floor_price

    def set_ad_slot_floor_price(self, ad_slot_floor_price):
        self.ad_slot_floor_price = ad_slot_floor_price

    def get_creative_id(self):
        return self.creative_id

    def set_creative_id(self, creative_id):
        self.creative_id = creative_id

    def get_advertiser_id(self):
        return self.advertiser_id

    def set_advertiser_id(self, advertiser_id):
        self.advertiser_id = advertiser_id

    def get_user_tags(self):
        return self.user_tags

    def set_user_tags(self, user_tags):
        self.user_tags = user_tags

    def set_all_values(self, line):
        fields = line.strip().split("\t")
        self.bid_id = fields[0]
        self.timestamp = fields[1]
        self.visitor_id = fields[3]
        self.user_agent = fields[4]
        self.ip_address = fields[5]
        self.region = fields[6]
        self.city = fields[7]
        self.ad_exchange = fields[8]
        self.domain = fields[9]
        self.url = fields[10]
        self.anonymous_url_id = fields[11]
        self.ad_slot_id = fields[12]
        self.ad_slot_width = fields[13]
        self.ad_slot_height = fields[14]
        self.ad_slot_visibility = fields[15]
        self.ad_slot_format = fields[16]
        self.ad_slot_floor_price = fields[17]
        self.creative_id = fields[18]
        self.advertiser_id = fields[19]
        self.user_tags = fields[20]