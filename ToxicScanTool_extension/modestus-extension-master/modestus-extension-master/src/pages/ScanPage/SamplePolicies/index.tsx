import { Button, Input, Textarea } from "@nextui-org/react";
import { twMerge } from "tailwind-merge";
import { Fragment, useCallback, useState } from "react";
import { PolicyItem } from "../../../types/moderate.types";
import { ERROR_MESSAGES, SAMPLE_POLICIES } from "../../../constants/moderate";
import { ResetFilledIcon } from "../../../components/Icons/ResetIcon";
import { CloseFilledIcon } from "../../../components/Icons/CloseIcon";
import { CheckFilledIcon } from "../../../components/Icons/CheckIcon";
import { PlusFilledIcon } from "../../../components/Icons/PlusIcon";
import LaunchMethod from "../../LaunchMethod";

type Props = {
  policies: PolicyItem[];
  handleUpdatePolicies: (newPolicies: PolicyItem[]) => void;
};

const SamplePolicies = ({ policies, handleUpdatePolicies }: Props) => {
  const [isFocus, setIsFocus] = useState(false);

  const updatePolicy = (policy: any, changes: any) => ({
    ...policy,
    ...changes,
  });

  const onPolicyChange = useCallback(
    (key: string, value: string | boolean, policyIdx: number) => {
      const newPolicies = policies.map((policy: PolicyItem, index) => {
        let errorMsg = "";
        if (index === policyIdx && policy.errorMsg) {
          if (!policy.newLabel) {
            errorMsg = ERROR_MESSAGES.policy_name;
          } else if (!policy.newValue) {
            errorMsg = ERROR_MESSAGES.policy_description;
          } else if (policy.isEdit) {
            errorMsg = ERROR_MESSAGES.policy_not_saved;
          } else {
            errorMsg = "";
          }
        }
        return index === policyIdx
          ? updatePolicy(policy, {
              isEdit: true,
              isFocus: false,
              errorMsg,
              [key]: value,
            })
          : policy;
      });
      handleUpdatePolicies(newPolicies);
    },
    [policies, handleUpdatePolicies],
  );

  const saveChromeLocalStorage = (policies: PolicyItem[]) => {
    chrome.storage.local.set({
      moderation: {
        policies,
      },
    });
  };

  const onSavePolicy = (policyIdx: number) => {
    const newPolicies = policies.map((policy: PolicyItem, index) =>
      index === policyIdx
        ? updatePolicy(policy, {
            isEdit: false,
            isFocus: false,
            errorMsg: "",
            label: policy.newLabel,
            value: policy.newValue,
          })
        : policy,
    );
    handleUpdatePolicies(newPolicies);
    saveChromeLocalStorage(newPolicies);
  };

  const onResetPolicy = (policyIdx: number) => {
    const newPolicies = policies.map((policy: PolicyItem, index) => {
      return index === policyIdx
        ? updatePolicy(policy, {
            isEdit: false,
            errorMsg: "",
            newLabel: policy.label,
            newValue: policy.value,
          })
        : policy;
    });
    handleUpdatePolicies(newPolicies);
  };

  const onAddNewPolicy = () => {
    const newPolicy = {
      label: "",
      value: "",
      newLabel: "",
      newValue: "",
      isEdit: false,
      isFocus: true,
      errorMsg: "",
    };
    handleUpdatePolicies([...policies, newPolicy]);
  };

  const onDeletePolicy = (policyIdx: number) => {
    const newPolicies = policies.filter((_, index) => index !== policyIdx);
    handleUpdatePolicies(newPolicies);
    saveChromeLocalStorage(newPolicies);
  };

  const onResetPolices = () => {
    handleUpdatePolicies(SAMPLE_POLICIES);
    saveChromeLocalStorage(SAMPLE_POLICIES);
  };

  return (
    <div className="flex-1 space-y-6" onBlur={() => setIsFocus(false)}>
      <div className="flex items-center justify-between">
        <h6 className="text-10 font-medium text-gray-400">DEFAULT POLICIES</h6>
        <div className="flex items-center gap-2">
          <Button
            className="hover:effect-scale h-6 min-h-6 w-6 min-w-6 rounded-full border border-gray-800 bg-divider p-1"
            onClick={onResetPolices}
          >
            <ResetFilledIcon />
          </Button>
          <LaunchMethod />
        </div>
      </div>
      <div className="flex flex-col items-center gap-3">
        {policies.map((policy, index) => (
          <div key={index} className="group relative w-full">
            <div
              className={twMerge(
                "w-full flex-1 space-y-2 rounded-xl border border-transparent bg-divider p-4",
                policy.errorMsg &&
                  "border-error-500 transition-all duration-300 ease-linear",
              )}
            >
              <div className="flex items-center justify-between">
                <Input
                  classNames={{
                    inputWrapper:
                      "!bg-transparent pl-0 min-h-5 h-5 group-data-[focus-visible=true]:ring-0 group-data-[focus-visible=true]:ring-offset-0",
                    input: "placeholder:text-gray-400 !text-white text-[12px]",
                  }}
                  placeholder="Policy name"
                  value={policy.newLabel}
                  onValueChange={(value) =>
                    onPolicyChange("newLabel", value, index)
                  }
                  autoFocus={policy.isFocus}
                  onKeyUp={(e) => e.key === "Enter" && onSavePolicy(index)}
                  onFocus={() => setIsFocus(true)}
                />
                <div className="flex items-center gap-1">
                  <div
                    className="hover:effect-scale cursor-pointer p-1"
                    onClick={() => {
                      if (!policy.isEdit) {
                        return onDeletePolicy(index);
                      }
                      onResetPolicy(index);
                    }}
                  >
                    <CloseFilledIcon />
                  </div>
                  <div
                    className={twMerge(
                      "hover:effect-scale cursor-pointer p-1",
                      !policy.isEdit && "hidden",
                    )}
                    onClick={() => onSavePolicy(index)}
                  >
                    <CheckFilledIcon />
                  </div>
                </div>
              </div>
              <Textarea
                classNames={{
                  inputWrapper:
                    "!bg-transparent p-0 min-h-4 h-4 group-data-[focus-visible=true]:ring-0 group-data-[focus-visible=true]:ring-offset-0",
                  input: "placeholder:text-gray-400 !text-gray-400 text-[12px]",
                }}
                placeholder="Add description..."
                value={policy.newValue}
                onValueChange={(value) =>
                  onPolicyChange("newValue", value, index)
                }
                onKeyUp={(e) => e.key === "Enter" && onSavePolicy(index)}
                onFocus={() => setIsFocus(true)}
                maxRows={2}
              />
            </div>
            {policy.errorMsg && (
              <span className="w-full text-left text-12 leading-none text-error-500">
                {policy.errorMsg}
              </span>
            )}
            <div
              className={twMerge(
                "invisible absolute -top-[26px] left-1/4 items-center justify-center rounded-lg bg-gray-800 px-3 py-1 text-[10px] text-white opacity-0 transition-all duration-200 ease-in-out",
                index === 0 &&
                  !isFocus &&
                  "group-hover:visible group-hover:opacity-100",
              )}
            >
              Click to customize policy
            </div>
          </div>
        ))}
        {policies.length < 3 && (
          <Button
            className="hover:effect-scale h-10 min-h-10 w-10 min-w-10 rounded-full border border-gray-800 bg-divider p-1"
            onClick={onAddNewPolicy}
          >
            <PlusFilledIcon />
          </Button>
        )}
      </div>
    </div>
  );
};

export default SamplePolicies;
