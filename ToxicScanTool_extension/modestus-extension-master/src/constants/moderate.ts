import { CheckAndAppendReasoning } from "../types/moderate.types";
import { scanRedditPage } from "../utils/contentRedditScanner";
import { scanXPage } from "../utils/contentTweetScanner";

const createSamplePoliciesItem = (label: string, value: string) => ({
  label,
  value,
  newLabel: label,
  newValue: value,
  isEdit: false,
  isFocus: false,
  errorMsg: "",
});

export const SAMPLE_POLICIES = [
  createSamplePoliciesItem(
    "Clickbait",
    "The content uses misleading or sensationalized text to entice clicks, such as like or retweet for reward.",
  ),
  createSamplePoliciesItem(
    "Sexuality",
    "The content contains inappropriate sexual content or language that may be explicit, suggestive, or offensive.",
  ),
  createSamplePoliciesItem(
    "Toxicity",
    "The content contains abusive, hateful, or discriminatory language towards individuals or groups.",
  ),
];

export const ERROR_MESSAGES = {
  policy_name: "Policy name cannot be empty.",
  policy_description: "Policy description cannot be empty.",
  policy_not_saved: "Policy has not been saved.",
};

export const URLS_SCAN: any = {
  "x.com": scanXPage,
  "reddit.com": scanRedditPage,
  "reddit.com/comments": scanRedditPage,
};

export const SCAN_PAGE_STATUS = {
  LOADING_STATUS: "LOADING_STATUS",
  UPDATE_MODERATION: "UPDATE_MODERATION",
  START_SCAN: "START_SCAN",
  UPDATE_AUTO_SCAN: "UPDATE_AUTO_SCAN",
  REQ_AUTO_SCAN_STATUS: "REQ_AUTO_SCAN_STATUS",
  AUTO_SCAN_STATUS: "AUTO_SCAN_STATUS",
};

export const CONTENT_LOADED: CheckAndAppendReasoning["CONTENT_LOADED"] = {};
