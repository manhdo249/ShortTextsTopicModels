import { Button, useDisclosure } from "@nextui-org/react";
import { useEffect, useState } from "react";
import SelectMethodModal from "./SelectMethodModal";

export type MethodType = "popup" | "sidePanel";

const LaunchMethod = () => {
  const { isOpen, onClose, onOpen } = useDisclosure();
  const [method, setMethod] = useState<MethodType>("popup");

  useEffect(() => {
    chrome.storage.local.get("panelMethod", (result) => {
      if (result.panelMethod) {
        setMethod(result.panelMethod);
      }
    });
  }, []);

  const onMethodChange = (value: MethodType) => {
    const selectedMethod = value;
    setMethod(selectedMethod);

    chrome.storage.local.set({ panelMethod: selectedMethod });

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      tabs.forEach((tab) => {
        if (tab.id) {
          if (selectedMethod === "sidePanel") {
            window.close();
            chrome.sidePanel.setOptions({
              path: "sidepanel.html",
              enabled: true,
            });
            chrome.sidePanel.open({
              windowId: tab.windowId,
            });
            chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
          } else {
            chrome.action.openPopup();
            chrome.sidePanel.setOptions({
              enabled: false,
            });
            chrome.sidePanel.setPanelBehavior({
              openPanelOnActionClick: false,
            });
          }
        }
      });
    });
  };

  return (
    <>
      <Button
        className="hover:effect-scale h-6 min-h-6 w-6 min-w-6 rounded-full border border-gray-800 bg-divider p-1"
        onClick={onOpen}
      >
        <svg
          viewBox="0 0 256 256"
          width="20"
          height="20"
          style={{
            fill: "white",
          }}
        >
          <path d="M216 40H40a16 16 0 0 0-16 16v144a16 16 0 0 0 16 16h176a16 16 0 0 0 16-16V56a16 16 0 0 0-16-16Zm0 16v40H40V56ZM40 112h56v88H40Zm176 88H112v-88h104v88Z" />
        </svg>
      </Button>

      <SelectMethodModal
        isOpen={isOpen}
        onClose={onClose}
        method={method}
        onMethodChange={onMethodChange}
      />
    </>
  );
};

export default LaunchMethod;
