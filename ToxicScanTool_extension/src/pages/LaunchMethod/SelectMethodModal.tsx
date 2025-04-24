import { Modal, ModalBody, ModalContent, ModalHeader } from "@nextui-org/react";
import React from "react";
import { CheckFilledIcon } from "../../components/Icons/CheckIcon";
import { twMerge } from "tailwind-merge";
import { MethodType } from ".";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  method: MethodType;
  onMethodChange: (method: MethodType) => void;
}

const METHOD_LIST = [
  {
    name: "Popup",
    desc: "Open Modestus at the top right corner",
    icon: () => (
      <svg
        viewBox="0 0 256 256"
        width="24"
        height="24"
        style={{
          fill: "white",
        }}
      >
        <path d="M216 40H40a16 16 0 0 0-16 16v144a16 16 0 0 0 16 16h176a16 16 0 0 0 16-16V56a16 16 0 0 0-16-16Zm0 160H40V56h176v144ZM80 84a12 12 0 1 1-12-12 12 12 0 0 1 12 12Zm40 0a12 12 0 1 1-12-12 12 12 0 0 1 12 12Z"></path>
      </svg>
    ),
    key: "popup",
  },
  {
    name: "Side panel",
    desc: "Open Modestus on the browser’s Side Panel and stick it there",
    icon: () => (
      <svg
        viewBox="0 0 256 256"
        width="24"
        height="24"
        style={{
          fill: "white",
        }}
      >
        <path d="M216 40H40a16 16 0 0 0-16 16v144a16 16 0 0 0 16 16h176a16 16 0 0 0 16-16V56a16 16 0 0 0-16-16ZM40 152h16a8 8 0 0 0 0-16H40v-16h16a8 8 0 0 0 0-16H40V88h16a8 8 0 0 0 0-16H40V56h40v144H40Zm176 48H96V56h120v144Z"></path>
      </svg>
    ),
    key: "sidePanel",
  },
];

const SelectMethodModal: React.FC<Props> = ({
  isOpen,
  onClose,
  method,
  onMethodChange,
}) => {
  return (
    <Modal
      size="md"
      isOpen={isOpen}
      onClose={onClose}
      classNames={{
        base: "bg-gray-950 border border-gray-800 rounded-3xl m-0 max-w-[320px]",
        header:
          "border-b border-b-divider text-[16px] font-medium p-4 text-white",
        body: "p-4",
        footer: "pt-0",
        closeButton: "mt-[10px] mr-2 hover:bg-gray-800 text-white",
      }}
      placement="center"
    >
      <ModalContent>
        <ModalHeader>Choose your open method</ModalHeader>
        <ModalBody>
          <div className="overflow-hidden rounded-3xl bg-gray-1000">
            {METHOD_LIST.map((item) => (
              <div
                key={item.key}
                className={twMerge(
                  "flex cursor-pointer items-center justify-between p-4 transition-all duration-300 ease-linear first:border-b first:border-b-divider hover:bg-gray-800",
                  method === item.key && "bg-gray-800",
                )}
                onClick={() => onMethodChange(item.key as MethodType)}
              >
                <div className="flex items-center gap-3">
                  <div>{item.icon()}</div>
                  <div className="space-y-1">
                    <span className="text-14 font-medium text-white">
                      {item.name}
                    </span>
                    <p
                      className={twMerge(
                        "max-w-[150px] text-12 text-gray-400",
                        item.key === "sidePanel" && "max-w-[190px]",
                      )}
                    >
                      {item.desc}
                    </p>
                  </div>
                </div>
                {method === item.key ? (
                  <div className="flex h-4 w-4 items-center justify-center rounded-full bg-aqua">
                    <CheckFilledIcon color="black" />
                  </div>
                ) : (
                  <></>
                )}
              </div>
            ))}
          </div>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};

export default SelectMethodModal;
