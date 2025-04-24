import { useEffect, useState } from "react";
import { checkIcon } from "../assets/icons";
import {
  LoadingResponse,
  NotificationScanProps,
} from "../types/moderate.types";
import { SCAN_PAGE_STATUS } from "../constants/moderate";
import { twMerge } from "tailwind-merge";

const NotiScan = () => {
  const [noti, setNoti] = useState<{
    isVisible: boolean;
    type: NotificationScanProps | string;
  }>({
    isVisible: false,
    type: "",
  });

  useEffect(() => {
    const handleMessage = (message: LoadingResponse) => {
      if (message.type === SCAN_PAGE_STATUS.LOADING_STATUS) {
        setNoti((prevState) => ({
          ...prevState,
          isVisible: true,
          type: message.scanStatus,
        }));
      }
    };

    chrome.runtime.onMessage.addListener(handleMessage);

    return () => {
      chrome.runtime.onMessage.removeListener(handleMessage);
    };
  }, []);

  useEffect(() => {
    if (noti.isVisible) {
      setTimeout(() => {
        setNoti((prevState) => ({
          ...prevState,
          isVisible: false,
          type: "",
        }));
      }, 1000);
    }
  }, [noti.isVisible]);

  const bgColor = noti.type === "error" ? "bg-red-400" : "bg-[#64D5E4]";
  const messageText = noti.type === "error" ? "Scan error!" : "Scanned";
  const icon =
    noti.type === "success" ? <img src={checkIcon} alt="check icon" /> : null;

  return (
    <div
      className={twMerge(
        "invisible absolute -top-12 left-1/2 -translate-x-1/2 opacity-0",
        noti.isVisible &&
          "visible top-12 opacity-100 transition-all duration-500 ease-in-out",
      )}
    >
      <div
        className={twMerge("flex items-center rounded-full px-2 py-1", bgColor)}
      >
        {icon}
        <span className="text-[14px] font-medium text-black">
          {messageText}
        </span>
      </div>
    </div>
  );
};

export default NotiScan;
