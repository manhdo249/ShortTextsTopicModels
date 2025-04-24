import { CONTENT_LOADED } from "../constants/moderate";
import { moderateContent } from "../services/moderate";
import { ModerationState } from "../types/moderate.types";
import {
  checkAndAppendReasoning,
  hashContent,
  removeContentElements,
  scanWrapperElement,
  showHideElement,
  toggleButtonElement,
} from "./common";

// Function to get the tweet content and its corresponding HTML elements
const getContent = () => {
  const elements = document.querySelectorAll<HTMLElement>(
    '[data-testid="tweet"]',
  );

  // Convert NodeList to an array and process each tweet element
  return Array.from(elements).map((element) => {
    const clone = element.cloneNode(true) as HTMLElement;

    // Remove any previously added moderation results
    clone
      .querySelectorAll(".moderation-result")
      .forEach((resultElement) => resultElement.remove());

    // Extract the tweet text
    const tweetText =
      clone.querySelector<HTMLElement>('[data-testid="tweetText"]')
        ?.innerText || "";

    return {
      text: tweetText,
      element,
    };
  });
};

// Function to append moderation reasoning to the tweet element
const appendReasoning = (
  hash: string,
  scanResults: {
    [key: string]: {
      reasoning: string;
    };
  },
  elementsMap: Map<string, HTMLElement>,
) => {
  const element = elementsMap.get(hash);
  if (!element) return;

  // Select the media and tweet text elements within the tweet
  const mediaElement = element.querySelector<HTMLElement>(
    ".css-175oi2r.r-9aw3ui",
  );
  const tweetTextElement = element.querySelector<HTMLElement>(
    '[data-testid="tweetText"]',
  );
  const toggleBtnElement = element.querySelector<HTMLElement>(".toggle-button");

  if (tweetTextElement) {
    // Ensure the data-visibility attribute is set
    if (!tweetTextElement.hasAttribute("data-visibility")) {
      tweetTextElement.setAttribute("data-visibility", "hidden");
    }

    const currentVisibility = tweetTextElement.getAttribute("data-visibility");
    const displayNone = currentVisibility === "hidden" ? "none" : "";
    const resultKeys = Object.keys(scanResults);

    if (resultKeys.length) {
      // Hide the tweet text and media if there are scan results
      tweetTextElement.style.display = displayNone;
      if (mediaElement) {
        mediaElement.style.display = displayNone;
      }

      // Insert moderation results before the tweet text if not already present
      let resultElement =
        element.querySelector<HTMLElement>(".moderation-result");
      if (!resultElement) {
        resultElement = scanWrapperElement(scanResults);
        tweetTextElement.insertAdjacentElement("beforebegin", resultElement);
      }

      // Add a toggle button to show/hide the tweet text and media
      if (!toggleBtnElement) {
        const { showElement, hideElement } = showHideElement();
        const button = toggleButtonElement(
          currentVisibility === "hidden" ? hideElement : showElement,
        );

        button.addEventListener("click", () => {
          const newVisibility =
            tweetTextElement.getAttribute("data-visibility") === "hidden"
              ? "visible"
              : "hidden";

          tweetTextElement.setAttribute("data-visibility", newVisibility);

          const display = newVisibility === "hidden" ? "none" : "";
          button.innerHTML =
            newVisibility === "hidden" ? hideElement : showElement;

          tweetTextElement.style.display = display;
          if (mediaElement) {
            mediaElement.style.display = display;
          }
        });
        resultElement.appendChild(button);
      }
    } else {
      // If there are no scan results, restore the tweet text and media visibility
      removeContentElements(element);
      tweetTextElement.removeAttribute("data-visibility");
      tweetTextElement.style.display = "";
      if (mediaElement) {
        mediaElement.style.display = "";
      }
    }
  }
};

// Main function to scan the X (formerly Twitter) page for tweet content
export const scanXPage = async (
  apiKey: string,
  moderation: ModerationState,
  scannedContentHashes?: Set<string>,
  autoScan?: boolean,
) => {
  try {
    const getContentResult = getContent();
    const elementsMap = new Map<string, HTMLElement>();

    // Hash each tweet content and map it to its element
    getContentResult.forEach((item) => {
      const contentHash = hashContent(item.text);
      elementsMap.set(contentHash, item.element);
    });

    if (getContentResult.length > 0) {
      // If there are already scanned content hashes, filter out previously scanned content
      if (scannedContentHashes) {
        const moderationScanned = getContentResult.filter((item) => {
          const contentHash = hashContent(item.text);
          if (scannedContentHashes.has(contentHash)) {
            return false;
          } else {
            scannedContentHashes.add(contentHash);
            return true;
          }
        });

        if (moderationScanned.length === 0) {
          // If no new content to scan, append reasoning for the already scanned content
          getContentResult.forEach((item) =>
            checkAndAppendReasoning({
              CONTENT_LOADED,
              item,
              elementsMap,
              appendReasoning,
            }),
          );
        } else {
          if (autoScan) {
            // If autoScan is enabled, moderate new content
            let moderationPromises = moderationScanned.map((item) => {
              if (
                checkAndAppendReasoning({
                  CONTENT_LOADED,
                  item,
                  elementsMap,
                  appendReasoning,
                })
              ) {
                return Promise.resolve(null);
              }

              return moderateContent(
                apiKey,
                item.text,
                moderation.policies,
              ).then((result) => {
                if (result) {
                  const contentHash = hashContent(item.text);
                  CONTENT_LOADED[contentHash] = result;
                  return result;
                }
                return null;
              });
            });

            // Wait for all moderation results and then append reasoning
            const moderationResults = await Promise.all(moderationPromises);
            moderationResults.forEach((result, index) => {
              if (result !== null) {
                const contentHash = hashContent(moderationScanned[index].text);
                appendReasoning(contentHash, result, elementsMap);
              }
            });
          }
        }
      } else {
        // If no scanned content hashes, scan all content
        if (autoScan) {
          const moderationPromises = getContentResult.map((item) => {
            return moderateContent(apiKey, item.text, moderation.policies).then(
              (result) => {
                if (result) {
                  const contentHash = hashContent(item.text);
                  CONTENT_LOADED[contentHash] = result;
                  return result;
                }
                return null;
              },
            );
          });

          // Wait for all moderation results and then append reasoning
          const moderationResults = await Promise.all(moderationPromises);
          moderationResults.forEach((result, index) => {
            if (result !== null) {
              const contentHash = hashContent(getContentResult[index].text);
              appendReasoning(contentHash, result, elementsMap);
            }
          });
        }
      }
    } else {
      console.log("No content found or content is empty.");
    }
  } catch (error) {
    console.error("Error in scanPage:", error);
  }
};
