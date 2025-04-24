import axios from "axios";
import { ENDPOINTS } from "../configs/apiEndpoints";
import { convertPoliciesArrayToObject } from "../helpers/moderateHelper";
import { PolicyItem } from "../types/moderate.types";
import { v4 as uuidv4 } from "uuid";

// const API_KEY = "e9c3185e-d0d8-49ba-a3da-9d2b4947c15a";

export const getApiKeyModerate = async () => {
  try {
    const response = await axios.get(ENDPOINTS.apiKey);
    return response.data;
  } catch (error) {
    console.error("Error fetching API Key Moderate:", error);
  }
};

export const postModerate = async ({
  apiKey,
  content,
  policies,
}: {
  apiKey: string;
  content: string;
  policies: PolicyItem[];
}) => {
  const response = await fetch(ENDPOINTS.moderate, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "moev-api-key": apiKey,
    },
    body: JSON.stringify({
      request_id: uuidv4(),
      content,
      metrics: convertPoliciesArrayToObject(policies),
    }),
  });

  if (response.ok) {
    const data = await response.json();
    return data.result;
  } else {
    console.error("Error submitting moderation:", response.status);
    return null;
  }
};

export const moderateContent = async (
  apiKey: string,
  content: string,
  policies: PolicyItem[],
) => {
  try {
    const moderateData = { apiKey, content, policies };
    return await postModerate(moderateData);
  } catch (e) {
    console.error("Something went wrong!", e);
    return null;
  }
};
