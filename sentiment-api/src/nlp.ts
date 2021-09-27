//@ts-ignore
import aposToLexForm from "apos-to-lex-form";
//@ts-ignore
import { WordTokenizer, SentimentAnalyzer, PorterStemmer } from "natural";
//@ts-ignore
import SpellCorrector from "spelling-corrector";
//@ts-ignore
import stopword from "stopword";

const tokenizer = new WordTokenizer();
const spellCorrector = new SpellCorrector();

const analyzer = new SentimentAnalyzer("English", PorterStemmer, "afinn");
export function getSentiment(str: string): -1 | 0 | 1 {
  if (!str.trim()) {
    return 0;
  }
  const lexed = aposToLexForm(str)
    .toLowercase()
    .replace(/[^a-zA-z\s]+/g, "");

  const tokenized = tokenizer.tokenize(lexed);
  const fixedSpelling = tokenized.map((word) => spellCorrector.correct(word));
  const stopWordsRemoved = stopword.removeStopWords(fixedSpelling);
  const analyzed = analyzer.getSentiment(stopWordsRemoved);
  if (analyzed >= 1) return 1;
  if (analyzed === 0) return 0;
  return -1;
}
