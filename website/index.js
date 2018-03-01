var reviewerName = document.getElementById("name");
var reviewerID = document.getElementById("id");
var overall = document.getElementById("overall");
var summary = document.getElementById("summary");
var reviewTime = document.getElementById("time");
var unixReviewTime = document.getElementById("unix-time");
var reviewText = document.getElementById("text");
var helpful = document.getElementById("helpful");
var pluralHandler = document.getElementById("plural-handler");
var unhelpful = document.getElementById("unhelpful");
var demo = document.getElementById("demo");
var predict = document.getElementById("predict");
var output = document.getElementById("output");

var demoData = {
	"reviewerID": "A2NYK9KWFMJV4Y",
	"asin": "B0002E5518",
	"reviewerName": "Mike Tarrani \"Jazz Drummer\"",
	"helpful": [1, 1],
	"reviewText": "One thing I love about this extension bar is it will fit over the relatively large diameter down tubes on my hi-hat stands. I use anOn Stage Microphone 13-inch Gooseneckto connect to the bar, then I attach either aNady DM70 Drum and Instrument Microphoneor DM80 microphone. The bar-gooseneck arrangement is sturdy enough for that mic model.This also works well for mounting on microphone stands. I use it and a shorter gooseneck to mount the DM70 for tenor and alto saxophones, or a DM80 for baritones. Again, it works perfectly for my situations.I always keep a few of these, plus various size goosenecks, just in case I need to mount an additional microphone and I am short on stands. It's one more tool to make set up less stressful.Trust me, when you need one (and chances are you will if you're a drummer or a sound tech) you will thank yourself for having the foresight to purchase it for situations I cited above and those that I have not foreseen.",
	"overall": 5.0,
	"summary": "Highly useful - especially for drummers (and saxophonists)",
	"unixReviewTime": 1370822400,
	"reviewTime": "06 10, 2013"
};

function padTime(time) {
	if (time < 10) {
		return "0" + time;
	}
	return time;
}

function formatTimeForDB(time) {
	var date = new Date(time + " 00:00:00");
	var month = date.getMonth() + 1;
	var day = date.getDate();
	var year = date.getFullYear();
	var unix = Math.floor(date.getTime() / 1000);
	return {
		"reviewTime": padTime(month) + " " + padTime(day) + ", " + year,
		"unixReviewTime": unix
	};
}

function formatTimeForForm(time) {
	var date = new Date(time);
	var month = date.getMonth() + 1;
	var day = date.getDate();
	var year = date.getFullYear();
	return year + "-" + padTime(month) + "-" + padTime(day);
}

function equalDates(time1, time2) {
	var date1 = new Date(time1);
	var date2 = new Date(time2);
	if (date1.getMonth() == date2.getMonth()) {
		if (date1.getDate() == date2.getDate()) {
			if (date1.getFullYear() == date2.getFullYear()) {
				return true;
			}
		}
	}
	return false;
}

function autocomplete(n) {
	if (n > 0) {
		if (reviewerName.value.length < demoData.reviewerName.length) {
			reviewerName.value += demoData.reviewerName[reviewerName.value.length];
		}
		else if (reviewerID.value.length < demoData.reviewerID.length) {
			reviewerID.value += demoData.reviewerID[reviewerID.value.length];
		}
		else if (overall.dataset.rating != demoData.overall) {
			overall.dataset.rating = demoData.overall;
		}
		else if (!equalDates(formatTimeForDB(reviewTime.value).unixReviewTime, demoData.unixReviewTime)) {
			reviewTime.value = formatTimeForForm(demoData.unixReviewTime * 1000);
			unixReviewTime.innerHTML = demoData.unixReviewTime;
		}
		else if (summary.value.length < demoData.summary.length) {
			summary.value += demoData.summary[summary.value.length];
		}
		else if (reviewText.value.length < demoData.reviewText.length) {
			reviewText.value += demoData.reviewText[reviewText.value.length];
		}
		else if (parseInt(helpful.value) != demoData.helpful[0]) {
			helpful.value = demoData.helpful[0];
			helpfulListener();
		}
		else if (parseInt(unhelpful.value) != demoData.helpful[1] - demoData.helpful[0]) {
			unhelpful.value = demoData.helpful[1] - demoData.helpful[0];
		}
		requestAnimationFrame(function () {
			autocomplete(n - 1)
		});
	}
}

function numberListener(element) {
	element.value = element.value.replace(/[^\d]/g, "");
}

function helpfulListener() {
	numberListener(helpful);
	if (helpful.value == 1) {
		pluralHandler.innerHTML = "person";
	}
	else {
		pluralHandler.innerHTML = "people";
	}
}

function init() {
	overall.addEventListener("click", function (e) {
		var starsOffset = (5 - overall.dataset.rating) * 19;
		var totalOffset = e.offsetX - starsOffset;
		overall.dataset.rating = Math.floor(Math.min(Math.max(0, 1 + totalOffset / 19), 5));
	});
	helpful.addEventListener("input", helpfulListener);
	unhelpful.addEventListener("input", function () {
		numberListener(unhelpful);
	});
	reviewTime.addEventListener("input", function () {
		unixReviewTime.innerHTML = formatTimeForDB(reviewTime.value).unixReviewTime;
	});
	var autocompleter = function () {
		autocomplete(17);
	};
	demo.addEventListener("click", function () {
		if (demo.dataset.active == "true") {
			demo.dataset.active = "false";
			demo.innerHTML = "Activate Demo"
			window.removeEventListener("keydown", autocompleter);
		}
		else {
			demo.dataset.active = "true";
			demo.innerHTML = "Deactivate Demo"
			window.addEventListener("keydown", autocompleter);
		}
	})
	predict.addEventListener("click", function () {
		var xhr = new XMLHttpRequest();
		xhr.open("POST", "predict", true);
		xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
		xhr.onload = function () {
			output.innerHTML = this.responseText;
		};
		xhr.onerror = function (e) {
			console.log("Error: " + e);
		};
		var time = formatTimeForDB(reviewTime.value);
		xhr.send("review=" + JSON.stringify({
			"reviewerID": reviewerID.value,
			"asin": "0",
			"reviewerName": reviewerName.value,
			"helpful": [parseInt(helpful.value), parseInt(helpful.value) + parseInt(helpful.value)],
			"reviewText": reviewText.value,
			"overall": parseInt(overall.dataset.rating),
			"summary": summary.value,
			"unixReviewTime": time.unixReviewTime,
			"reviewTime": time.reviewTime
		}));
	});
}

init();
