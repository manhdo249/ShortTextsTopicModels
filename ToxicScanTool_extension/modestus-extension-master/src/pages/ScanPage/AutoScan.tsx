import { Checkbox } from "@nextui-org/react";
import { useLayoutEffect, useState } from "react";
import { SCAN_PAGE_STATUS } from "../../constants/moderate";

const { UPDATE_AUTO_SCAN } = SCAN_PAGE_STATUS;

const AutoScan = () => {
  const [autoScan, setAutoScan] = useState<boolean>(true);

  useLayoutEffect(() => {
    chrome.storage.local.get("autoScan", (res) => {
      if (typeof res["autoScan"] === "boolean") {
        setAutoScan(res["autoScan"]);
      }
    });
  }, []);

  const onAutoScanChange = (isSelected: boolean) => {
    chrome.runtime.sendMessage({
      type: UPDATE_AUTO_SCAN,
      autoScan: isSelected,
    });
    setAutoScan(isSelected);
  };

  return (
    <Checkbox
      color="primary"
      classNames={{
        base: "p-0 m-0",
        label: "text-white text-[12px] font-medium",
        wrapper:
          "bg-transparent group-data-[selected=true]:!bg-aqua group-data-[hover=true]:!bg-aqua before:!border-[1px] before:!border-aqua group-data-[selected=true]:before:!border-aqua rounded-md before:rounded-md after:rounded-md",
      }}
      onValueChange={onAutoScanChange}
      isSelected={autoScan}
    >
      Auto scan
    </Checkbox>
  );
};

export default AutoScan;
