/* HTML Elements */

var csrfmiddlewaretoken = document.getElementsByName("csrfmiddlewaretoken")[0];
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

/* Time Parsing */

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

/* Demo Feature */

var demoDatum = [
	/* Musical Instruments */
	{
		"reviewerID": "A2NYK9KWFMJV4Y",
		"asin": "B0002E5518",
		"reviewerName": "Mike Tarrani \"Jazz Drummer\"",
		"helpful": [1, 1],
		"reviewText": "One thing I love about this extension bar is it will fit over the relatively large diameter down tubes on my hi-hat stands. I use anOn Stage Microphone 13-inch Gooseneckto connect to the bar, then I attach either aNady DM70 Drum and Instrument Microphoneor DM80 microphone. The bar-gooseneck arrangement is sturdy enough for that mic model.This also works well for mounting on microphone stands. I use it and a shorter gooseneck to mount the DM70 for tenor and alto saxophones, or a DM80 for baritones. Again, it works perfectly for my situations.I always keep a few of these, plus various size goosenecks, just in case I need to mount an additional microphone and I am short on stands. It's one more tool to make set up less stressful.Trust me, when you need one (and chances are you will if you're a drummer or a sound tech) you will thank yourself for having the foresight to purchase it for situations I cited above and those that I have not foreseen.",
		"overall": 5.0,
		"summary": "Highly useful - especially for drummers (and saxophonists)",
		"unixReviewTime": 1370822400,
		"reviewTime": "06 10, 2013"
	},
	/* Baby */
	{
		"reviewerID": "ALRN58JO86V5E",
		"asin": "9729375011",
		"reviewerName": "John Ramahlo Jr.",
		"helpful": [2, 3], "reviewText":
		"THis has been helpful in tracking the feedings, diaper changes, naps and other developmental things for our little one but I REALLY wish they had laid it out so that you could leave it open and have the left side be AM and the right PM.  The way the pages are set up you have to flip a page to see a full day (starts with PM on the left and A.M. on the right).  It's a nitpick but if they had just let the user fill in AM or PM then the pages would be identical and you could look at a full day at a glance (with your hands full, holding a baby).  Sort of a pain to have to lean forward to pick it up and flip a page over just to see what happend prior.",
		"overall": 4.0,
		"summary": "ONe simple thing would improve it.",
		"unixReviewTime": 1355011200,
		"reviewTime": "12 9, 2012"
	},
	/* Video Games */
	{
		"reviewerID": "A2N4P35FVAGYAL",
		"asin": "9861019731",
		"reviewerName": "desorbod",
		"helpful": [2, 2],
		"reviewText": "Not sure what you can really write about this item, it works fine... I use it on the Wii to play Gamecube games and it works great and fast, but the Gamecube was always much faster than PS2 anyway...",
		"overall": 5.0,
		"summary": "This is what it says it is!",
		"unixReviewTime": 1363737600,
		"reviewTime": "03 20, 2013"
	},
	/* Office Products */
	{
		"reviewerID": "A1CBNUBPZPWH5D",
		"asin": "B00004Z5QP",
		"reviewerName": "Boston Lesbian \"Happily Married in Massachusetts\"",
		"helpful": [0, 0],
		"reviewText": "What can I say about address labels?  These worked well in the laser printer, didn't jam and peeled off cleanly.  The type on the labels printed clearly too.  These will work well for return address labels.  They're too small for regular addresses.  As usual Avery has a quality product.",
		"overall": 5.0,
		"summary": "Works well",
		"unixReviewTime": 1297900800,
		"reviewTime": "02 17, 2011"
	},
	/* Pet Supplies */
	{
		"reviewerID": "ALWWS8QBYN80B",
		"asin": "B00005MF9T",
		"reviewerName": "B. Novick \"honest user\"",
		"helpful": [6, 10],
		"reviewText": "I have one female cat that weighs under 10 pounds and is about 15 years old.  I had used EverClean for years and really had no complaints.  Last year our cat, Poloma, started to urinate more frequently and my back was starting to hurt when I bent down to clean the clumps.I had seen the LitterMaid litter box advertised for several years and had always wondered if it could really be true and work as it was advertised.  Sorry, I finally gave in and purchased along with the tent top.The tent top came with several pieces missing/broken.  I called the company who asked me to fax the part numbers so they could forward them to me.  They sent me the wrong parts and I have not been able to get around to ask for them again.  I will though.The machine seems to work to not on the clumps of urine.  I have to now clean the clumps out DAILY rather than every few days.  The machine tries but is too weak to push,lift and dump.  INstead I walk into the room only to see the red blinking light which indicates a jam.I called the company and explained my problem.  The only advice they could give me was to be sure and use a quality clumping litter.  I would have though the high price I pay for Litter Green would have been considered quality clumping litter.  I am now looking for my fouth \"quality litter\" that might work with my single cat.As usual, if it looks too good to be true, it probably is.",
		"overall": 1.0,
		"summary": "Sorry I finally gave in.",
		"unixReviewTime": 1100649600,
		"reviewTime": "11 17, 2004"
	}
];
var demoData;

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

/* Event Listeners */

function reviewerIDListener() {
	reviewerID.value = reviewerID.value.toUpperCase().slice(0, 14);
}

function overallListener(mouse) {
	var starsOffset = (5 - overall.dataset.rating) * 19;
	var totalOffset = mouse.offsetX - starsOffset;
	overall.dataset.rating = Math.floor(Math.min(Math.max(0, 1 + totalOffset / 19), 5));
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

function unhelpfulListener() {
	numberListener(unhelpful);
}

function reviewTimeListener() {
	unixReviewTime.innerHTML = formatTimeForDB(reviewTime.value).unixReviewTime;
}

function autocompleter() {
	autocomplete(17);
}

function demoListener() {
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
}

var prediction = "Books";
function predictListener() {
	document.getElementById(prediction).className = "";
	var xhr = new XMLHttpRequest();
	xhr.open("POST", location, true);
	xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
	xhr.setRequestHeader("X-CSRFToken", csrfmiddlewaretoken.value);
	xhr.onload = function () {
		var prediction = this.responseText;
		document.getElementById(prediction).className = "predicted";
	};
	xhr.onerror = function (e) {
		console.log("Error: " + e);
	};
	var time = formatTimeForDB(reviewTime.value);
	xhr.send("review=" + encodeURIComponent(JSON.stringify({
		"reviewerID": reviewerID.value,
		"asin": "0",
		"reviewerName": reviewerName.value,
		"helpful": [parseInt(helpful.value), parseInt(helpful.value) + parseInt(helpful.value)],
		"reviewText": reviewText.value,
		"overall": parseInt(overall.dataset.rating),
		"summary": summary.value,
		"unixReviewTime": time.unixReviewTime,
		"reviewTime": time.reviewTime
	})));
}

/* Initialization */

function init() {
	demoData = demoDatum[Math.floor(demoDatum.length * Math.random())];
	reviewerID.addEventListener("input", reviewerIDListener);
	overall.addEventListener("click", overallListener);
	helpful.addEventListener("input", helpfulListener);
	unhelpful.addEventListener("input", unhelpfulListener);
	reviewTime.addEventListener("input", reviewTimeListener);
	demo.addEventListener("click", demoListener);
	predict.addEventListener("click", predictListener);
}

init();
