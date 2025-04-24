import { v4 as uuidv4 } from "uuid";
import { CheckAndAppendReasoning } from "../types/moderate.types";
import { caretDownIcon, eyeNoneIcon, eyeOpenIcon } from "../assets/icons";
import { modestusLogoImg } from "../assets/images";

export const getChunks = (arr: Array<any>, chunkSize: number) => {
  if (chunkSize <= 0) throw "Invalid chunk size";
  const R = [];
  const len = arr.length;
  for (let i = 0; i < len; i += chunkSize) R.push(arr.slice(i, i + chunkSize));
  return R;
};

export const sleep = (ms: number) => {
  return new Promise<void>((res, rej) => setTimeout(() => res(), ms));
};

export const currentTimestamp = () => {
  return Math.round(new Date().getTime() / 1000);
};

export const isNumberInRange = (n: number, nStart: number, nEnd: number) => {
  return n >= nStart && n < nEnd ? true : false;
};

export const newUUID = () => {
  return uuidv4();
};

export const capitalizeFirstLetter = (str: string): string => {
  if (str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
  return "";
};

export const hashContent = (content: string): string => {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0;
  }
  return hash.toString();
};

export const checkAndAppendReasoning = ({
  CONTENT_LOADED,
  item,
  elementsMap,
  appendReasoning,
}: CheckAndAppendReasoning) => {
  const contentHash = hashContent(item.text);
  if (CONTENT_LOADED[contentHash]) {
    const scanResults = CONTENT_LOADED[contentHash];
    if (scanResults) {
      appendReasoning(contentHash, scanResults, elementsMap);
    }
    return true;
  }
  return false;
};

export const contentScanElement = (className: string, reasoning: string) => {
  const p = document.createElement("p");
  p.className = className;
  p.innerText = reasoning;
  p.style.backgroundColor = "#AFEBF2";
  p.style.borderRadius = "6px";
  p.style.border = "1px solid #1F4A5A14";
  p.style.padding = "4px";
  p.style.fontSize = "12px";
  p.style.color = "#0F303D";
  p.style.opacity = "1";
  p.style.fontFamily = "sans-serif";
  return p;
};

export const toggleButtonElement = (content: string) => {
  const div = document.createElement("div");
  div.className = `toggle-button`;
  div.innerHTML = content;
  div.style.position = "absolute";
  div.style.right = "32px";
  div.style.top = "10.5px";
  div.style.zIndex = "20";
  div.style.width = "16px";
  div.style.height = "16px";
  div.style.cursor = "pointer";

  return div;
};

export const showHideElement = () => {
  const showElement = `<img src=${eyeOpenIcon} style="margin: 0px; width: 16px; height: 16px;"/>`;
  const hideElement = `<img src=${eyeNoneIcon} style="margin: 0px; width: 16px; height: 16px;"/>`;
  return { showElement, hideElement };
};

export const scanWrapperElement = (scanResults: any) => {
  const scanKeys = Object.keys(scanResults);

  const div = document.createElement("div");
  div.className = "moderation-result";
  div.style.position = "relative";
  div.style.zIndex = "3";
  div.style.marginBottom = "8px";
  div.innerHTML = `
    <div style="background-color: #D5F5F8; border-radius: 12px; border: 1px solid #3AC1D64D; padding: 8px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center; gap: 6px;">
          <img src="${modestusLogoImg}" alt="Modestus" style="width: 18px; height: 18px; margin: 0px;" />
          <span style="font-size: 12px; color: #123237; font-weight: bold; font-family: 'Inter', sans-serif;">Detected</span>
        </div>
        <button type="button" id="caretIcon" style="width: 16px; height: 16px; background-color: transparent; border: none; display: flex; justify-content: center; align-items: center; padding: 0px; margin: 0px; cursor: pointer; transform: rotate(180deg);">
          <img src="${caretDownIcon}" style="margin: 0px; width: 16px; height: 16px;"/>
        </button>
      </div>
      <div id="scanContent" style="display: block; margin-top: 6px;">
        ${scanKeys
          .map((scanKey) => {
            return `
              <p key="${scanKey}" style="font-size: 12px; color: #123237; font-family: 'Inter', sans-serif; margin: 0px;">
                <span style="font-weight: bold;">${capitalizeFirstLetter(scanKey)}:</span> ${scanResults[scanKey]?.reasoning}
              </p>
            `;
          })
          .join("")}
      </div>
    <div>
  `;

  // Get references to the caret icon and the content container
  const caretIcon = div.querySelector<HTMLElement>("#caretIcon");
  const scanContent = div.querySelector<HTMLElement>("#scanContent");

  // Add click event listener to toggle the visibility
  if (caretIcon) {
    caretIcon.addEventListener("click", () => {
      if (scanContent) {
        const isContentVisible = scanContent.style.display === "block";
        scanContent.style.display = isContentVisible ? "none" : "block";

        // Optionally, you can rotate the caret icon when the content is visible
        caretIcon.style.transform = isContentVisible
          ? "rotate(0deg)"
          : "rotate(180deg)";
      }
    });
  }

  return div;
};

export const removeContentElements = (element: HTMLElement) => {
  element.querySelector(".moderation-result")?.remove();
  element.querySelector<HTMLElement>(".toggle-button")?.remove();
};

export const checkUrlSupport = (url: string): boolean => {
  if (url) {
    return (
      url.includes("x.com") ||
      url.includes("twitter.com") ||
      url.includes("reddit.com")
    );
  }
  return false;
};
