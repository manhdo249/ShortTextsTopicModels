import { useLayoutEffect, useState } from "react";
import { modestusLogo } from "../../assets/icons";
import { checkUrlSupport } from "../../utils/common";

const OnlySupportModal = () => {
  const [isUrlSupport, setIsUrlSupport] = useState<boolean>(true);

  const queryChromeTabs = () => {
    chrome.tabs.query(
      {
        active: true,
        currentWindow: true,
      },
      (tabs) => {
        tabs.forEach((tab) => {
          if (tab.url) {
            const isSupported = checkUrlSupport(tab.url);
            setIsUrlSupport(isSupported);
          }
        });
      },
    );
  };

  useLayoutEffect(() => {
    queryChromeTabs();
    chrome.tabs.onActivated.addListener((activeInfo) => {
      chrome.tabs.get(activeInfo.tabId, (tab) => {
        if (chrome.runtime.lastError) {
          console.error(chrome.runtime.lastError.message);
        } else {
          if (tab.url) {
            const isSupported = checkUrlSupport(tab.url);
            setIsUrlSupport(isSupported);
          }
        }
      });
    });

    chrome.tabs.onUpdated.addListener(() => {
      queryChromeTabs();
    });
  }, []);

  return !isUrlSupport ? (
    <div className="fixed inset-0 z-[99] bg-[#12131A52] backdrop-blur">
      <div className="flex h-full w-full items-center justify-center">
        <div className="flex flex-col items-center gap-6 rounded-2xl border border-gray-800 bg-gray-950 px-6 py-10">
          <img src={modestusLogo} className="h-7 w-10" />
          <div className="space-y-2">
            <h5 className="text-center text-14 font-medium text-white">
              Welcome to Modestus
            </h5>
            <p className="max-w-[186px] text-center text-12 text-gray-400">
              The current version of Modestus only supports scanning on{" "}
              <span className="text-white">x.com and reddit.com</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  ) : (
    <></>
  );
};

export default OnlySupportModal;
