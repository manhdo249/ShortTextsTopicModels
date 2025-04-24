import {
  SAMPLE_POLICIES,
  SCAN_PAGE_STATUS,
  URLS_SCAN,
} from "./constants/moderate";
import { getApiKeyModerate } from "./services/moderate";
import { ScanAction } from "./types/moderate.types";

console.log("content script loaded!");

const { START_SCAN, LOADING_STATUS, AUTO_SCAN_STATUS } = SCAN_PAGE_STATUS;
const scannedContentHashes = new Set<string>();

let moderateKey = "";
let debounceTimeout: number | null = null;
let isScanning: boolean = false;
let autoScan: boolean = true;

chrome.storage.local.get("autoScan", (res) => {
  if (typeof res["autoScan"] === "boolean") {
    autoScan = res["autoScan"] ? true : false;
  }
});

const fetchApiKey = async () => {
  moderateKey = await getApiKeyModerate();
};

const getScanCurrentSite = () => {
  const url = window.location.href;
  const urlKey = Object.keys(URLS_SCAN).find((key) => url.includes(key));
  if (urlKey) {
    return URLS_SCAN[urlKey];
  }
  return "";
};

const scanPage = getScanCurrentSite();

if (scanPage) {
  fetchApiKey();
}

const handleScanPage = async () => {
  if (!isScanning && moderateKey) {
    isScanning = true;

    chrome.storage.local.get("moderation", async (res) => {
      const moderation = res["moderation"] || {
        policies: SAMPLE_POLICIES,
      };
      if (moderation) {
        await scanPage(moderateKey, moderation, scannedContentHashes, autoScan);
      }
    });
    isScanning = false;
  }
};

const handleScroll = () => {
  if (debounceTimeout) {
    clearTimeout(debounceTimeout);
  }

  debounceTimeout = window.setTimeout(async () => {
    if (!isScanning && moderateKey) {
      handleScanPage();
    }
  }, 1);
};

chrome.runtime.onMessage.addListener(
  async (message: ScanAction, sender, sendResponse) => {
    try {
      if (message.action === START_SCAN) {
        if (moderateKey) {
          await scanPage(moderateKey, message.moderation, undefined, true);
          chrome.runtime.sendMessage({
            type: LOADING_STATUS,
            isLoading: false,
            scanStatus: "success",
          });
          sendResponse({ status: "Page scanned successfully!" });
        } else {
          chrome.runtime.sendMessage({
            type: LOADING_STATUS,
            isLoading: false,
            scanStatus: "error",
          });
        }
      }
    } catch (e) {
      chrome.runtime.sendMessage({
        type: LOADING_STATUS,
        isLoading: false,
        scanStatus: "error",
      });
      sendResponse({ status: "Page scan error!" });
    }
  },
);

chrome.runtime.onMessage.addListener(async (message) => {
  if (message.type === AUTO_SCAN_STATUS) {
    autoScan = message.autoScan;
  }
});

window.addEventListener("scroll", handleScroll);
