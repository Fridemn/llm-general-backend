<template>
    <div>
      <h1>实时语音助手</h1>
      
      <!-- 连接设置 -->
      <div class="card">
        <h2>连接设置</h2>
        <div class="config-row">
          <label for="api-url">WebSocket API地址:</label>
          <input type="text" id="api-url" v-model="apiUrl" />
        </div>
        <div class="btn-group">
          <button class="neutral" @click="connectWebSocket" :disabled="isConnected">
            连接服务器
          </button>
          <button class="secondary" @click="disconnectWebSocket" :disabled="!isConnected">
            断开连接
          </button>
        </div>
        <div class="status" :class="connectionStatusClass">{{ connectionStatusText }}</div>
      </div>
  
      <!-- 助手配置 -->
      <div class="card">
        <h2>助手配置</h2>
        <div class="config-panel">
          <div class="config-row">
            <label for="model">LLM模型:</label>
            <select id="model" v-model="selectedModel" @change="updateParams">
              <option v-for="model in availableModels" :key="model" :value="model">
                {{ model }}
              </option>
            </select>
          </div>
          <div class="config-row">
            <label for="history-id">对话历史ID:</label>
            <input type="text" id="history-id" v-model="historyId" @change="updateParams" />
            <button class="neutral" @click="createNewHistory" :disabled="!isConnected">
              创建新对话
            </button>
          </div>
        </div>
      </div>
  
      <!-- 语音交互 -->
      <div class="card">
        <h2>实时语音交互</h2>
        <div class="btn-group">
          <button class="primary" @click="startVoiceChat" :disabled="startBtnDisabled">
            开始实时对话
          </button>
          <button class="secondary" @click="stopVoiceChat" :disabled="!isVoiceChatting">
            停止对话
          </button>
        </div>
        <div v-if="isVoiceChatting" class="voice-status">
          <div class="status-text">正在聆听...</div>
          <div class="voice-indicator" :class="{ active: isSpeaking }"></div>
        </div>
        <div class="status">{{ statusText }}</div>
      </div>
  
      <!-- 对话历史 -->
      <div class="card">
        <h2>对话历史</h2>
        <div class="conversation">
          <div v-for="(message, index) in conversationHistory" 
               :key="index"
               :class="message.sender === 'user' ? 'user-message' : 'assistant-message'">
            {{ message.text }}
          </div>
        </div>
      </div>
  
      <!-- 音频输出部分增强 -->
      <div class="card">
        <h2>音频输出</h2>
        <div class="audio-queue">
          <audio ref="audioPlayer" 
                 @ended="onAudioEnded" 
                 @error="onAudioError"
                 @canplay="onAudioCanPlay"
                 style="display: none;">
          </audio>
          <div v-if="isPlaying" class="playing-status">
            <div class="audio-progress">
              <div class="progress-bar">
                <div class="progress" :style="{ width: audioProgressWidth }"></div>
              </div>
            </div>
            正在播放: {{ currentAudio ? currentAudio.text : '语音响应' }}
          </div>
          <div v-if="audioQueue.length > 0 && !isPlaying" class="queue-status">
            队列中: {{ audioQueue.length }} 个音频片段
          </div>
        </div>
      </div>
  
      <!-- 日志 -->
      <div class="card">
        <h2>日志</h2>
        <div class="log">
          <div v-for="(logEntry, index) in logs" :key="index">
            {{ logEntry }}
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import { ref, onMounted, computed, onUnmounted } from 'vue';
  
  export default {
    name: 'RealtimeVoiceAssistant',
    setup() {
      // 修改默认WebSocket URL
      const apiUrl = ref('ws://localhost:8000/ws/realtime-voice-chat');
      
      // 基础状态
      const socket = ref(null);
      const isConnected = ref(false);
      const isVoiceChatting = ref(false);
      const isSpeaking = ref(false);
      const clientId = ref(null);
      const historyId = ref('');
      const selectedModel = ref('');
      const availableModels = ref([]);
      const audioQueue = ref([]);
      const currentAudio = ref(null);
      const audioPlayer = ref(null);
      const isPlaying = ref(false); // 添加此行，解决isPlaying未定义错误
      const audioProgress = ref(0); // 添加此行，用于音频进度跟踪
  
      // 界面状态
      const statusText = ref('等待连接...');
      const connectionStatusText = ref('未连接');
      const connectionStatusClass = ref('disconnected');
      const conversationHistory = ref([]);
      const logs = ref([]);
  
      // 计算属性
      const startBtnDisabled = computed(() => {
        return !isConnected.value || !historyId.value || !selectedModel.value || isVoiceChatting.value;
      });
      
      // 添加音频进度宽度计算属性
      const audioProgressWidth = computed(() => {
        return audioProgress.value + '%';
      });
  
      // 添加重连控制
      const maxReconnectAttempts = ref(3);
      const reconnectCount = ref(0);
      const reconnectDelay = ref(1000); // 初始延迟1秒
  
      const connectWebSocket = async () => {
        try {
          if (socket.value) {
            socket.value.close();
            await new Promise(resolve => setTimeout(resolve, 500));
            socket.value = null;
          }
  
          updateConnectionStatus('正在连接...', 'processing');
          addToLog('正在连接服务器...');
  
          socket.value = new WebSocket(apiUrl.value);
          
          // 设置较短的连接超时
          const connectionTimeout = setTimeout(() => {
            if (socket.value && socket.value.readyState !== WebSocket.OPEN) {
              socket.value.close();
              handleConnectionError('连接超时');
            }
          }, 5000);
  
          socket.value.onopen = () => {
            clearTimeout(connectionTimeout);
            isConnected.value = true;
            reconnectCount.value = 0;
            updateConnectionStatus('已连接', 'connected');
            addToLog('WebSocket连接已建立');
            fetchAvailableModels();
            startHeartbeat();
          };
  
          socket.value.onclose = handleDisconnect;
          socket.value.onerror = handleConnectionError;
          socket.value.onmessage = handleServerMessage;
  
        } catch (error) {
          handleConnectionError(error);
        }
      };
  
      const disconnectWebSocket = () => {
        if (socket.value) {
          socket.value.close();
        }
      };
  
      const handleDisconnect = (event) => {
        stopHeartbeat();
        isConnected.value = false;
        isVoiceChatting.value = false;
        isSpeaking.value = false;
        
        const reason = event.reason || '未知原因';
        updateConnectionStatus('未连接', 'disconnected');
        addToLog(`WebSocket连接已关闭 - 代码: ${event.code}, 原因: ${reason}`);
        
        // 如果不是用户主动关闭，尝试重连
        if (event.code !== 1000 && reconnectCount.value < maxReconnectAttempts.value) {
          reconnectCount.value++;
          const delay = reconnectDelay.value * reconnectCount.value;
          addToLog(`将在 ${delay/1000} 秒后尝试重连(${reconnectCount.value}/${maxReconnectAttempts.value})...`);
          
          setTimeout(() => {
            connectWebSocket();
          }, delay);
        }
      };
  
      const handleConnectionError = (error) => {
        addToLog(`连接错误: ${error}`);
        updateConnectionStatus('连接失败', 'disconnected');
        
        // 尝试重连
        if (reconnectCount.value < maxReconnectAttempts.value) {
          reconnectCount.value++;
          const delay = reconnectDelay.value * reconnectCount.value;
          addToLog(`将在 ${delay/1000} 秒后尝试重连(${reconnectCount.value}/${maxReconnectAttempts.value})...`);
          
          setTimeout(() => {
            connectWebSocket();
          }, delay);
        }
      };
  
      // 心跳检测
      let heartbeatInterval = null;
      const startHeartbeat = () => {
        stopHeartbeat();
        heartbeatInterval = setInterval(() => {
          if (socket.value && socket.value.readyState === WebSocket.OPEN) {
            socket.value.send(JSON.stringify({
              type: 'ping',
              timestamp: Date.now()
            }));
          } else {
            stopHeartbeat();
          }
        }, 10000);  // 更频繁的心跳检测
      };
  
      const stopHeartbeat = () => {
        if (heartbeatInterval) {
          clearInterval(heartbeatInterval);
          heartbeatInterval = null;
        }
      };
  
      // 语音聊天功能
      const startVoiceChat = () => {
        if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
          addToLog('WebSocket未连接');
          return;
        }
  
        socket.value.send(JSON.stringify({
          command: 'start',
          server_url: 'ws://127.0.0.1:8765'
        }));
  
        isVoiceChatting.value = true;
        updateStatus('开始实时语音对话...');
      };
  
      const stopVoiceChat = () => {
        if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
          return;
        }
  
        socket.value.send(JSON.stringify({
          command: 'stop'
        }));
  
        isVoiceChatting.value = false;
        isSpeaking.value = false;
        updateStatus('已停止语音对话');
      };
  
      // 服务器消息处理
      const handleServerMessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'connection':
              clientId.value = message.client_id;
              addToLog(`连接成功，客户端ID: ${clientId.value}`);
              break;
  
            case 'start':
              if (message.status === 'success') {
                addToLog('实时语音对话已启动');
              }
              break;
  
            case 'stop':
              if (message.status === 'success') {
                addToLog('实时语音对话已停止');
              }
              break;
  
            case 'recognition_result':
              handleRecognitionResult(message);
              break;
  
            case 'llm_stream_chunk':
              handleLLMStreamChunk(message);
              break;
  
            case 'llm_response':
              handleLLMResponse(message);
              break;
  
            case 'error':
              handleError(message);
              break;
  
            case 'tts_sentence_complete':
              handleTTSSentenceComplete(message);
              break;
          }
        } catch (error) {
          console.error('处理消息错误:', error);
          addToLog(`处理消息错误: ${error}`);
        }
      };
  
      // 消息处理函数
      const handleRecognitionResult = (message) => {
        addMessageToConversation(message.text, 'user');
        updateStatus(`已识别: ${message.text}`);
        isSpeaking.value = false;
      };
  
      const handleLLMStreamChunk = (message) => {
        if (!document.querySelector('.assistant-response-building')) {
          addMessageToConversation('', 'assistant', 'assistant-response-building');
        }
        appendToLastMessage(message.content);
      };
  
      const handleLLMResponse = (message) => {
        if (message.audio_url) {
          playAudioResponse(message.audio_url);
        }
      };
  
      const handleError = (message) => {
        addToLog(`错误: ${message.message}`);
        updateStatus(`错误: ${message.message}`);
      };
  
      const handleTTSSentenceComplete = (message) => {
        if (message.audio_url) {
          const fullUrl = window.location.protocol + '//' + window.location.hostname + ':8000' + message.audio_url;
          addToLog(`收到TTS音频: ${message.text.substring(0, 20)}...`);
          
          // 添加到音频队列
          audioQueue.value.push({
            url: fullUrl,
            text: message.text
          });
          
          // 如果没有正在播放的音频，开始播放
          if (!isPlaying.value) {
            playNextAudio();
          }
        }
      };
  
      // 工具函数
      const updateConnectionStatus = (text, status) => {
        connectionStatusText.value = text;
        connectionStatusClass.value = status;
      };
  
      const updateStatus = (message) => {
        statusText.value = message;
      };
  
      const addToLog = (message) => {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        logs.value.push(`[${timeStr}] ${message}`);
      };
  
      const addMessageToConversation = (text, sender, className = '') => {
        conversationHistory.value.push({ text, sender, className });
      };
  
      const appendToLastMessage = (text) => {
        const lastMessage = conversationHistory.value.find(msg => msg.className === 'assistant-response-building');
        if (lastMessage) {
          lastMessage.text += text;
        }
      };
  
      // 音频播放相关
      const playAudioResponse = (url) => {
        const fullUrl = window.location.protocol + '//' + window.location.hostname + ':8000' + url;
        audioQueue.value.push({
          url: fullUrl,
          text: '语音回复'
        });
        
        if (!currentAudio.value) {
          playNextAudio();
        }
      };
  
      const playNextAudio = () => {
        if (audioQueue.value.length > 0 && !isPlaying.value) {
          isPlaying.value = true;
          currentAudio.value = audioQueue.value[0];
          audioProgress.value = 0;
          
          // 设置音频源并播放
          if (audioPlayer.value) {
            audioPlayer.value.src = currentAudio.value.url;
            audioPlayer.value.play().catch(error => {
              console.error('播放音频失败:', error);
              onAudioError(error);
            });
          }
        }
      };
  
      const onAudioEnded = () => {
        addToLog('音频播放完成');
        audioQueue.value.shift();
        currentAudio.value = null;
        isPlaying.value = false;
        
        // 继续播放队列中的下一个
        playNextAudio();
      };
  
      const onAudioCanPlay = () => {
        audioProgress.value = 0;
        // 启动进度更新
        startProgressUpdate();
      };
  
      // 启动进度更新
      let progressInterval = null;
      const startProgressUpdate = () => {
        if (progressInterval) {
          clearInterval(progressInterval);
        }
        
        progressInterval = setInterval(() => {
          if (audioPlayer.value && audioPlayer.value.duration) {
            audioProgress.value = (audioPlayer.value.currentTime / audioPlayer.value.duration) * 100;
          }
        }, 100);
      };
  
      const onAudioError = (error) => {
        addToLog(`音频播放错误: ${error.message || '未知错误'}`);
        audioQueue.value.shift();
        currentAudio.value = null;
        isPlaying.value = false;
        
        // 尝试播放下一个
        playNextAudio();
      };
  
      // 模型和历史记录相关
      const fetchAvailableModels = async () => {
        try {
          const response = await fetch('http://localhost:8000/llm/available-models');
          if (response.ok) {
            const data = await response.json();
            availableModels.value = data['available-models'];
            selectedModel.value = availableModels.value[0] || '';
          }
        } catch (error) {
          addToLog(`获取模型列表失败: ${error}`);
        }
      };
  
      const createNewHistory = async () => {
        try {
          const response = await fetch('http://localhost:8000/llm/history/new', {
            method: 'POST'
          });
          if (response.ok) {
            const data = await response.json();
            historyId.value = data.history_id;
            addToLog(`创建了新对话，ID: ${historyId.value}`);
            updateParams();
          }
        } catch (error) {
          addToLog(`创建对话失败: ${error}`);
        }
      };
  
      // 优化参数更新
      const updateParams = () => {
        if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
          addToLog('WebSocket未连接，无法更新参数');
          return;
        }
  
        if (!selectedModel.value) {
          addToLog('警告: 未选择模型');
          return;
        }
  
        if (!historyId.value) {
          addToLog('警告: 未设置对话历史ID');
          return;
        }
  
        socket.value.send(JSON.stringify({
          command: 'set_params',
          model: selectedModel.value,
          history_id: historyId.value
        }));
  
        addToLog('已更新会话参数');
      };
  
      // 生命周期钩子
      onMounted(() => {
        addToLog('页面已加载，请点击"连接服务器"开始');
      });
  
      // 在组件卸载时清理资源
      onUnmounted(() => {
        stopHeartbeat();
        if (socket.value) {
          socket.value.close();
        }
        if (progressInterval) {
          clearInterval(progressInterval);
        }
      });
  
      return {
        // 状态
        isConnected,
        isVoiceChatting,
        isSpeaking,
        apiUrl,
        historyId,
        selectedModel,
        availableModels,
        statusText,
        connectionStatusText,
        connectionStatusClass,
        conversationHistory,
        logs,
        currentAudio,
        audioPlayer,
        startBtnDisabled,
        reconnectCount,
        maxReconnectAttempts,
        isPlaying,
        audioProgress,
        audioProgressWidth,
        audioQueue, // 添加这一行，修复 Cannot read properties of undefined (reading 'length') 错误
  
        // 方法
        connectWebSocket,
        disconnectWebSocket,
        startVoiceChat,
        stopVoiceChat,
        createNewHistory,
        updateParams,
        playNextAudio,
        onAudioEnded,
        onAudioCanPlay,
        onAudioError
      };
    }
  };
  </script>
  
  <style scoped>
  .card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .config-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
  }
  
  .btn-group {
    display: flex;
    gap: 10px;
    margin: 10px 0;
  }
  
  button {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
  }
  
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .neutral { background-color: #2196F3; color: white; }
  .primary { background-color: #4CAF50; color: white; }
  .secondary { background-color: #f44336; color: white; }
  
  .status {
    margin: 10px 0;
    padding: 10px;
    border-radius: 4px;
    text-align: center;
  }
  
  .connected { color: #4CAF50; }
  .disconnected { color: #f44336; }
  .processing { color: #ff9800; }
  
  .conversation {
    height: 300px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
  }
  
  .user-message,
  .assistant-message {
    margin: 5px 0;
    padding: 8px;
    border-radius: 4px;
  }
  
  .user-message {
    background-color: #E3F2FD;
    margin-left: auto;
    text-align: right;
  }
  
  .assistant-message {
    background-color: #F5F5F5;
  }
  
  .voice-status {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin: 10px 0;
  }
  
  .voice-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #ccc;
  }
  
  .voice-indicator.active {
    background-color: #4CAF50;
    animation: pulse 1s infinite;
  }
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
  }
  
  .log {
    height: 200px;
    overflow-y: auto;
    padding: 10px;
    background-color: #f5f5f5;
    font-family: monospace;
    border-radius: 4px;
  }
  
  .playing-status {
    padding: 10px;
    background-color: #E3F2FD;
    border-radius: 4px;
    margin: 5px 0;
  }
  
  .audio-progress {
    margin-bottom: 8px;
  }
  
  .progress-bar {
    height: 6px;
    background-color: #e0e0e0;
    border-radius: 3px;
    overflow: hidden;
  }
  
  .progress {
    height: 100%;
    background-color: #2196F3;
    transition: width 0.1s linear;
  }
  
  .queue-status {
    padding: 8px;
    color: #666;
    font-size: 0.9em;
    text-align: center;
  }
  </style>
  