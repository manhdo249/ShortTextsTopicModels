export interface PolicyItem {
  label: string;
  value: string;
  newLabel: string;
  newValue: string;
  isEdit: boolean;
  isFocus: boolean;
  errorMsg: string;
}

export interface ModerationState {
  policies: PolicyItem[];
}

export type NotificationScanProps = {
  type: "success" | "error";
};

export interface LoadingResponse {
  type: "LOADING_STATUS";
  isLoading: boolean;
  scanStatus: NotificationScanProps;
}

export interface ModerationResponse {
  type: "UPDATE_MODERATION";
  moderation: ModerationState;
}

export interface AutoScanResponse {
  type: "UPDATE_AUTO_SCAN";
  autoScan: boolean;
}

export interface ScanAction {
  action: "START_SCAN";
  moderation: ModerationState;
  querySelector: string;
}

export type MessageTypes = ModerationResponse | AutoScanResponse;

export interface CheckAndAppendReasoning {
  CONTENT_LOADED: {
    [hash: string]: {
      [key: string]: {
        reasoning: string;
      };
    };
  };
  item: { text: string; element: HTMLElement };
  elementsMap: Map<string, HTMLElement>;
  appendReasoning: (
    hash: string,
    scanResults: {
      [key: string]: {
        reasoning: string;
      };
    },
    elementsMap: Map<string, HTMLElement>,
  ) => void;
}
