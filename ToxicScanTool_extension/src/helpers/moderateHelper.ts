import { ERROR_MESSAGES } from "../constants/moderate";
import { ModerationState, PolicyItem } from "../types/moderate.types";

export const convertPoliciesArrayToObject = (
  data: PolicyItem[],
): { [key: string]: string } => {
  return data.reduce(
    (obj, item) => {
      obj[item.label] = item.value;
      return obj;
    },
    {} as { [key: string]: string },
  );
};

export const checkContentAndPolicies = (data: ModerationState) => {
  let hasError = false;

  for (const policy of data.policies) {
    if (!policy.newLabel) {
      policy.errorMsg = ERROR_MESSAGES.policy_name;
      hasError = true;
    } else if (!policy.newValue) {
      policy.errorMsg = ERROR_MESSAGES.policy_description;
      hasError = true;
    } else if (policy.isEdit) {
      policy.errorMsg = ERROR_MESSAGES.policy_not_saved;
      hasError = true;
    } else {
      policy.errorMsg = "";
    }
  }

  return { data, hasError };
};
