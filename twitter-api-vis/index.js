const Twit = require('twit');
const dotenv = require('dotenv')
const sentiment = require('multilang-sentiment');
const createCsvWriter = require('csv-writer').createObjectCsvWriter

dotenv.config();

const {
	CONSUMER_KEY, CONSUMER_SECRET,
	ACCESS_TOKEN, ACCESS_TOKEN_SECRET
} = process.env;


const config_twitter = {
	consumer_key : CONSUMER_KEY,
	consumer_secret : CONSUMER_SECRET,
	access_token : ACCESS_TOKEN,
	access_token_secret : ACCESS_TOKEN_SECRET,
	timeout_ms: 60*1000,
}

let api = new Twit(config_twitter);

let data = [];

const csvWriter = createCsvWriter({
	path: 'data.csv',
	header: [
		{id: 'datetime', title: 'datetime'},
		{id: 'location', title: 'location'},
		{id: 'score', title: 'score'},
		{id: 'words', title: 'words'},
		{id: 'txt', title: 'txt'}
	],
})

async function getTweets(q, count){
	try{
		let tweets = await api.get('search/tweets', {
			q, count, tweet_mode: 'extended',
		});

		tweets.data.statuses.map(tweet => {
			let score,
			datetime = tweet.created_at,
			location = tweet.retweeted_status ? tweet.retweeted_status.user.location.replace(`''`,'') : tweet.user.location.replace(`'`, ''),
			txt = tweet.retweeted_status ? tweet.retweeted_status.full_text : tweet.full_text
				.split(/ |\n/)
				.filter(v => !v.startsWith('http'))
				.join(' ');
		
		
      score = sentiment(txt, 'en');
      data.push({
        datetime: `${datetime}`,
        location: `${location}`,
        score: `${score.score}`,
        words: `${score.words}`,
        txt: `${txt}`,
      });
		return data;
		});

		csvWriter
			.writeRecords(data)
			.then(() => console.log('data file'))
	} catch(err){
		console.log('no')
	}
}

getTweets('iphone 13', 100)