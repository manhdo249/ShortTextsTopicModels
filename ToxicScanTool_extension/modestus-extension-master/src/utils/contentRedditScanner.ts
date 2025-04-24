import { CONTENT_LOADED } from "./../constants/moderate";
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

// Function to check if the current URL is a Reddit comments page
const redditCommentsUrl = () => {
  const url = typeof window !== "undefined" ? window.location.href : "";
  return url.includes("reddit.com") && url.includes("comments");
};

// Function to fetch and process content elements from the page
const getContent = () => {
  const isRedditComments = redditCommentsUrl();

  // Select elements based on the page type (post or comments)
  const elements = document.querySelectorAll<HTMLElement>(
    !isRedditComments ? "shreddit-post" : "shreddit-comment",
  );

  // Process each element and return an array of objects containing the text and original element
  return Array.from(elements).map((element) => {
    const clone = element.cloneNode(true) as HTMLElement;

    // Remove previous moderation results from the cloned element
    clone
      .querySelectorAll(".moderation-result")
      .forEach((resultElement) => resultElement.remove());

    // Extract the text content based on the page type (post or comment)
    const title =
      clone.querySelector<HTMLElement>(
        !isRedditComments ? '[slot="title"]' : '[slot="comment"]',
      )?.innerText || "";

    return {
      text: title,
      element,
    };
  });
};

// Function to append moderation reasoning to the content elements
const appendReasoning = (
  hash: string,
  scanResults: {
    [key: string]: {
      reasoning: string;
    };
  },
  elementsMap: Map<string, HTMLElement>,
) => {
  const isRedditComments = redditCommentsUrl();
  const element = elementsMap.get(hash);

  if (!element) return;

  // Select the relevant content elements within the DOM
  const titleElement = element.querySelector<HTMLElement>(
    !isRedditComments ? '[slot="title"]' : '[slot="comment"]',
  );
  const textBody = element.querySelector<HTMLElement>('[slot="text-body"]');
  const mediaElement = element.querySelector<HTMLElement>(
    '[slot="post-media-container"]',
  );
  const toggleBtnElement = element.querySelector<HTMLElement>(".toggle-button");

  if (titleElement) {
    if (!titleElement.hasAttribute("data-visibility")) {
      titleElement.setAttribute("data-visibility", "hidden");
    }

    const currentVisibility = titleElement.getAttribute("data-visibility");
    const displayNone = currentVisibility === "hidden" ? "none" : "";
    const resultKeys = Object.keys(scanResults);

    if (resultKeys.length) {
      // Hide or show content based on moderation results
      titleElement.style.display = displayNone;
      if (mediaElement) {
        mediaElement.style.display = displayNone;
      }
      if (textBody) {
        textBody.style.display = displayNone;
      }

      // Add moderation result elements to the DOM
      let resultElement =
        element.querySelector<HTMLElement>(".moderation-result");
      if (!resultElement) {
        resultElement = scanWrapperElement(scanResults);
        titleElement.insertAdjacentElement("beforebegin", resultElement);
      }

      // Add toggle button for visibility control
      if (!toggleBtnElement) {
        const { showElement, hideElement } = showHideElement();
        const button = toggleButtonElement(
          currentVisibility === "hidden" ? hideElement : showElement,
        );

        button.addEventListener("click", () => {
          const newVisibility =
            titleElement.getAttribute("data-visibility") === "hidden"
              ? "visible"
              : "hidden";

          titleElement.setAttribute("data-visibility", newVisibility);

          const display = newVisibility === "hidden" ? "none" : "";

          button.innerHTML =
            newVisibility === "hidden" ? hideElement : showElement;

          titleElement.style.display = display;
          if (mediaElement) {
            mediaElement.style.display = display;
          }
          if (textBody) {
            textBody.style.display = display;
          }
        });

        resultElement.appendChild(button);
      }
    } else {
      // Remove moderation elements if no results are found
      removeContentElements(element);
      titleElement.removeAttribute("data-visibility");
      titleElement.style.display = "";
      if (mediaElement) {
        mediaElement.style.display = "";
      }
      if (textBody) {
        textBody.style.display = "";
      }
    }
  }
};

// Main function to scan the Reddit page for moderation
export const scanRedditPage = async (
  apiKey: string,
  moderation: ModerationState,
  scannedContentHashes?: Set<string>,
  autoScan?: boolean,
) => {
  try {
    const getContentResult = getContent();
    const elementsMap = new Map<string, HTMLElement>();

    // Create a map of content hash and corresponding elements
    getContentResult.forEach((item) => {
      const contentHash = hashContent(item.text);
      elementsMap.set(contentHash, item.element);
    });

    if (getContentResult.length > 0) {
      // Divide the content into chunks for moderation
      const chunks = [];
      for (let i = 0; i < getContentResult.length; i += 10) {
        chunks.push(getContentResult.slice(i, i + 10));
      }

      for (const chunk of chunks) {
        if (scannedContentHashes) {
          // Filter out already scanned content
          const moderationScanned = chunk.filter((item) => {
            const contentHash = hashContent(item.text);
            if (scannedContentHashes.has(contentHash)) {
              return false;
            } else {
              scannedContentHashes.add(contentHash);
              return true;
            }
          });

          if (moderationScanned.length === 0) {
            // Append reasoning for already scanned content
            chunk.forEach((item) =>
              checkAndAppendReasoning({
                CONTENT_LOADED,
                item,
                elementsMap,
                appendReasoning,
              }),
            );
          } else if (autoScan) {
            // Perform moderation scan on new content
            const moderationPromises = moderationScanned.map((item) => {
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

            // Process and append moderation results
            const moderationResults = await Promise.all(moderationPromises);
            moderationResults.forEach((result, index) => {
              if (result !== null) {
                const contentHash = hashContent(moderationScanned[index].text);
                appendReasoning(contentHash, result, elementsMap);
              }
            });
          }
        } else if (autoScan) {
          // Perform moderation scan on all content if no scanned hashes are provided
          const moderationPromises = chunk.map((item) => {
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

          // Process and append moderation results
          const moderationResults = await Promise.all(moderationPromises);
          moderationResults.forEach((result, index) => {
            if (result !== null) {
              const contentHash = hashContent(chunk[index].text);
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
