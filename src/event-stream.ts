/**
 * Self-contained EventStream implementation.
 * Replaces the dependency on @mariozechner/pi-ai for stream handling.
 */

export class EventStream {
  isComplete: (event: unknown) => boolean;
  extractResult: (event: unknown) => unknown;
  queue: unknown[] = [];
  waiting: ((result: { value: unknown; done: boolean }) => void)[] = [];
  done = false;
  finalResultPromise: Promise<unknown>;
  resolveFinalResult!: (result: unknown) => void;

  constructor(
    isComplete: (event: unknown) => boolean,
    extractResult: (event: unknown) => unknown
  ) {
    this.isComplete = isComplete;
    this.extractResult = extractResult;
    this.finalResultPromise = new Promise((resolve) => {
      this.resolveFinalResult = resolve;
    });
  }

  push(event: unknown): void {
    if (this.done) return;
    if (this.isComplete(event)) {
      this.done = true;
      this.resolveFinalResult(this.extractResult(event));
    }
    const waiter = this.waiting.shift();
    if (waiter) {
      waiter({ value: event, done: false });
    } else {
      this.queue.push(event);
    }
  }

  end(result?: unknown): void {
    this.done = true;
    if (result !== undefined) {
      this.resolveFinalResult(result);
    }
    while (this.waiting.length > 0) {
      const waiter = this.waiting.shift()!;
      waiter({ value: undefined, done: true });
    }
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<unknown> {
    while (true) {
      if (this.queue.length > 0) {
        yield this.queue.shift()!;
      } else if (this.done) {
        return;
      } else {
        const result = await new Promise<{ value: unknown; done: boolean }>(
          (resolve) => this.waiting.push(resolve)
        );
        if (result.done) return;
        yield result.value;
      }
    }
  }

  result(): Promise<unknown> {
    return this.finalResultPromise;
  }
}

export class AssistantMessageEventStream extends EventStream {
  constructor() {
    super(
      (event: unknown) => {
        const e = event as { type?: string };
        return e.type === "done" || e.type === "error";
      },
      (event: unknown) => {
        const e = event as { type?: string; message?: unknown; error?: unknown };
        if (e.type === "done") return e.message;
        else if (e.type === "error") return e.error;
        throw new Error("Unexpected event type for final result");
      }
    );
  }
}

export function createAssistantMessageEventStream(): AssistantMessageEventStream {
  return new AssistantMessageEventStream();
}
