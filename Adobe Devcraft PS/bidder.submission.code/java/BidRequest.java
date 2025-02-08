package com.dtu.hackathon.bidding;

import java.io.Serializable;

public class BidRequest implements Serializable{

	private static final long serialVersionUID = -3012027079030559912L;

	private String bidId;
	private String timestamp ;
	private String visitorId;
	private String userAgent;
	private String ipAddress ;
	private String region;
	private String city;
	private String adExchange;
	private String domain ;
	private String url;
	private String anonymousURLID ;
	private String adSlotID;
	private String adSlotWidth;
	private String adSlotHeight;
	private String adSlotVisibility;
	private String adSlotFormat;
	private String adSlotFloorPrice;
	private String creativeID;
	private String advertiserId;
	private String userTags;

	public String getBidId() {
		return bidId;
	}

	public void setBidId(String bidId) {
		this.bidId = bidId;
	}

	public String getTimestamp() {
		return timestamp;
	}

	public void setTimestamp(String timestamp) {
		this.timestamp = timestamp;
	}

	public String getVisitorId() {
		return visitorId;
	}

	public void setVisitorId(String visitorId) {
		this.visitorId = visitorId;
	}

	public String getUserAgent() {
		return userAgent;
	}

	public void setUserAgent(String userAgent) {
		this.userAgent = userAgent;
	}

	public String getIpAddress() {
		return ipAddress;
	}

	public void setIpAddress(String ipAddress) {
		this.ipAddress = ipAddress;
	}

	public String getRegion() {
		return region;
	}

	public void setRegion(String region) {
		this.region = region;
	}

	public String getCity() {
		return city;
	}

	public void setCity(String city) {
		this.city = city;
	}

	public String getAdExchange() {
		return adExchange;
	}

	public void setAdExchange(String adExchange) {
		this.adExchange = adExchange;
	}

	public String getDomain() {
		return domain;
	}

	public void setDomain(String domain) {
		this.domain = domain;
	}

	public String getUrl() {
		return url;
	}

	public void setUrl(String url) {
		this.url = url;
	}

	public String getAnonymousURLID() {
		return anonymousURLID;
	}

	public void setAnonymousURLID(String anonymousURLID) {
		this.anonymousURLID = anonymousURLID;
	}

	public String getAdSlotID() {
		return adSlotID;
	}

	public void setAdSlotID(String adSlotID) {
		this.adSlotID = adSlotID;
	}

	public String getAdSlotWidth() {
		return adSlotWidth;
	}
	
	public void setAdSlotWidth(String adSlotWidth) {
		this.adSlotWidth = adSlotWidth;
	}

	public String getAdSlotHeight() {
		return adSlotHeight;
	}

	public void setAdSlotHeight(String adSlotHeight) {
		this.adSlotHeight = adSlotHeight;
	}

	public String getAdSlotVisibility() {
		return adSlotVisibility;
	}

	public void setAdSlotVisibility(String adSlotVisibility) {
		this.adSlotVisibility = adSlotVisibility;
	}

	public String getAdSlotFormat() {
		return adSlotFormat;
	}

	public void setAdSlotFormat(String adSlotFormat) {
		this.adSlotFormat = adSlotFormat;
	}

	public String getAdSlotFloorPrice() {
		return adSlotFloorPrice;
	}

	public void setAdSlotFloorPrice(String adSlotFloorPrice) {
		this.adSlotFloorPrice = adSlotFloorPrice;
	}

	public String getCreativeID() {
		return creativeID;
	}

	public void setCreativeID(String creativeID) {
		this.creativeID = creativeID;
	}

	public String getAdvertiserId() {
		return advertiserId;
	}

	public void setAdvertiserId(String advertiserId) {
		this.advertiserId = advertiserId;
	}

	public String getUserTags() {
		return userTags;
	}

	public void setUserTags(String userTags) {
		this.userTags = userTags;
	}

}