<template>
  <div class="omnicomplete-w">
    <input
      class="omni-input omni-shared"
      type="text"
      @keydown="keyPressed"
      v-model="inputValue"
      @keydown.tab.prevent="tabPressed"
      @keydown.enter.prevent="enterPressed"
      :placeholder="placeholder"
    />
    <input
      class="omni-completion omni-shared"
      v-model="completionsDisplay"
      :disabled="true"
      @click="focusInput"
    />
    <div class="gradient-div" :class="{ 'loading': loading }"></div>
  </div>
</template>
  
  <script lang="ts" setup>
  import { computed, nextTick, ref, toRefs, onMounted } from "vue";
  import { useDebounceFn } from "@vueuse/core";
  
  interface Props {
    completions?: string[];
    topic: string;
    loading: boolean;
  }
  
  const props = defineProps<Props>();
  const { completions, topic, loading } = toRefs(props);
  
  const emit = defineEmits([
    "get-autocomplete",
    "use-autocomplete",
    "clear-completions",
    "enter-pressed",
    "update-display-values",
  ]);
  
  const inputValue = ref("");
  const inputRef = ref<HTMLInputElement | null>(null);
  const displayValues = ref<string[]>([]); 
  
  const didAutocomplete = ref(false);
  
  const completion = computed(() => {
    return completionsDisplay.value || "";
  });
  
  const placeholder = computed(() => {
    return `Ask me anything about ${topic.value}`;
  });
  
  const completionsDisplay = computed<string>(() => {
    if (!completions.value) {
      return "";
    }
    const firstCompletion = completions.value[0];
    if (!firstCompletion) {
      return "";
    }
    return `${inputValue.value.trim()} ${firstCompletion}`;
  });
  
  function tabPressed() {
    const inputPreCompletion = inputValue.value;
    const completionSuffix = completions.value?.[0];
  
    inputValue.value = completion.value;
    didAutocomplete.value = true;  
    emit("use-autocomplete", inputPreCompletion, completionSuffix);

    nextTick(() => {
      inputRef.value?.scrollTo(inputRef.value.scrollWidth, 0);
    });
  }
  
  function enterPressed() {
    emit("enter-pressed", inputValue.value);
    displayValues.value.push(inputValue.value);
    emit("update-display-values", displayValues.value);
    inputValue.value = "";
    didAutocomplete.value = false;
    nextTick(scrollToBottom);
  }
  
  const keyPressedDebounce = useDebounceFn(function (_event: KeyboardEvent) {
    if (didAutocomplete.value) {
      return;
    }
    if (inputValue.value.length === 0) {
      didAutocomplete.value = false;
      return;
    }
    if (inputValue.value.split(" ").length < 3) {
      return;
    }
    emit("get-autocomplete", inputValue.value);
  }, 500);
  
  function keyPressed(event: KeyboardEvent) {
    emit("clear-completions");
    keyPressedDebounce(event);
    if (inputValue.value === "") {
      didAutocomplete.value = false;
    }
  }
  
  function focusInput() {
    inputRef.value?.focus();
  }

  function scrollToBottom() {
  const container = document.querySelector(".user-input-display-container");
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

onMounted(() => {
  nextTick(scrollToBottom);
});
</script>
  
  <style scoped>
  .omnicomplete-w {
    position: fixed;
    width: 75%;
    height: 7%;
    bottom: 5.5%;
    z-index: 1000;
  }
  
  .omni-input {
    width: 100%;
    height: 100%;
    position: relative;
    z-index: 1;
    background: rgb(8, 8, 8);
  }

 
  .omni-completion {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    border: 1px solid transparent;
    background: transparent;
    color: #fffd8280;
    z-index: 11;
    pointer-events: none;
    text-shadow: 0 0 10px yellow;
  }
  
  .omni-shared {
    border-radius: 10px;
    padding: 5px 15px;
    box-shadow: 0 0px 13px rgba(21, 59, 129, 0.822),
      0 0px 13px rgba(149, 19, 19, 0.769);
    border: none;
    outline: none;
    transition: box-shadow 0.3s ease-in-out, background 0.3s ease-in-out;
    font-size: 20px;
  }
  
  .omni-shared:focus {
    box-shadow: 0 0px 44px rgba(20, 77, 184, 0.98),
      0 0px 44px rgba(186, 15, 15, 0.95);
    background: rgb(0, 0, 0);
  }
  
  .gradient-div {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(45deg, #ff0000, #00ff00, #0000ff);
  background-size: 500% 500%;
  animation: gradientAnimation 10s ease infinite;
  border-radius: 50%;
  filter: blur(20px); /* Subtle blur for default state */
  z-index: -1;
  transition: opacity 0.3s ease-in-out, filter 0.3s ease-in-out;
  opacity: 0.1; /* Subtle opacity for default state */
}

.gradient-div.loading {
  width: 125%;
  height: 125%;
  opacity: 0.5; /* Increased opacity when loading */
  filter: blur(50px); /* Larger blur when loading */
  animation: gradientPulsate 3s ease infinite; /* Pulsating effect */
}

@keyframes gradientAnimation {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

@keyframes gradientPulsate {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.25); /* Pulsating effect */
  }
}
  </style>
  