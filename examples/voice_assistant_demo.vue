<template>
    <div>
      <h1>语音助手演示</h1>
  
      <div class="card">
        <h2>连接设置</h2>
        <div class="config-row">
          <label for="api-url">WebSocket API地址:</label>
          <input type="text" id="api-url" v-model="apiUrl" />
        </div>
        <div class="btn-group">
          <button id="connectBtn" class="neutral" @click="connectWebSocket" :disabled="connectBtnDisabled">连接服务器</button>
          <button id="disconnectBtn" class="secondary" @click="disconnectWebSocket" :disabled="disconnectBtnDisabled">断开连接</button>
        </div>
        <div id="connection-status" class="status" :class="connectionStatusClass">{{ connectionStatusText }}</div>
      </div>
  
      <div class="card">
        <h2>语音助手配置</h2>
        <div class="config-panel">
          <div class="config-row">
            <label for="model">LLM模型:</label>
            <select id="model" v-model="selectedModel" @change="setSessionParams">
              <option v-for="model in availableModels" :key="model" :value="model">
                {{ model }}
              </option>
            </select>
          </div>
          <div class="config-row">
            <label for="history-id">对话历史ID:</label>
            <input type="text" id="history-id" v-model="historyId" @change="onHistoryIdChange" />
            <button id="create-history-btn" class="neutral" @click="createNewHistory" :disabled="createHistoryBtnDisabled" style="margin-left: 10px;">创建新对话</button>
          </div>
          <div class="config-row">
            <label for="duration">录音时长 (秒):</label>
            <input type="number" id="duration" min="1" max="60" v-model.number="duration" />
          </div>
        </div>
      </div>
  
      <div class="card">
        <h2>TTS设置</h2>
        <div class="config-panel">
          <div class="config-row">
            <label for="provider">TTS提供者:</label>
            <select id="provider" v-model="provider" @change="setSessionParams">
              <option value="gsvi">GSVI</option>
            </select>
          </div>
          <div class="config-row">
            <label for="api-base">API基础URL:</label>
            <input type="text" id="api-base" v-model="apiBase" @change="setSessionParams" />
          </div>
          <div class="config-row">
            <label for="character">角色名称(可选):</label>
            <input type="text" id="character" v-model="character" placeholder="角色名称" @change="setSessionParams" />
          </div>
          <div class="config-row">
            <label for="emotion">情感(可选):</label>
            <input type="text" id="emotion" v-model="emotion" placeholder="情感" @change="setSessionParams" />
          </div>
        </div>
      </div>
  
      <div class="card">
        <h2>语音交互</h2>
        <div class="btn-group">
          <button id="startRecordBtn" class="primary" @click="startVoiceChat" :disabled="startRecordBtnDisabled">开始录音</button>
          <button id="stopRecordBtn" class="secondary" @click="stopRecording" :disabled="stopRecordBtnDisabled">停止录音</button>
        </div>
        <div class="progress-bar">
          <div id="progress" class="progress" :style="{ width: progressWidth }">{{ progressText }}</div>
        </div>
        <div id="statusDisplay" class="status">{{ statusText }}</div>
      </div>
  
      <div class="card">
        <h2>对话历史</h2>
        <div id="conversation" class="conversation">
          <div v-for="(message, index) in conversationHistory" :key="index" :class="message.sender === 'user' ? 'user-message' : 'assistant-message'">
            {{ message.text }}
          </div>
        </div>
      </div>
  
      <div class="card">
        <h2>音频输出</h2>
        <div class="audio-queue">
          <audio ref="audioPlayer"
                 @ended="onAudioEnded"
                 @error="onAudioError"
                 @loadstart="onAudioLoadStart"
                 @canplay="onAudioCanPlay"
                 style="display: none;">
          </audio>
          <div v-if="currentAudio" class="playing-status">
            正在播放: {{ currentAudio.text || '语音响应' }}
          </div>
        </div>
      </div>
  
      <div class="card">
        <h2>日志</h2>
        <div id="log" class="log">
          <div v-for="(logEntry, index) in logs" :key="index">
            {{ logEntry }}
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import { ref, onMounted, computed } from 'vue';
  
  export default {
    name: 'HomeView',
    setup() {
      // Reactive data
      const socket = ref(null);
      const clientId = ref(null);
      const isRecording = ref(false);
      const isConnected = ref(false);
      const isProcessingLlm = ref(false);
      const historyId = ref('');
      const availableModels = ref([]);
      const recordingInterval = ref(null);
      const currentStreamMessage = ref(null);
      const currentStreamBuffer = ref('');
      const heartbeatInterval = ref(null);
      const lastServerActivity = ref(Date.now());
  
      const apiUrl = ref('ws://localhost:8000/ws/voice-assistant');
      const duration = ref(10);
      const selectedModel = ref('');
      const provider = ref('gsvi');
      const apiBase = ref('http://127.0.0.1:5000');
      const character = ref('');
      const emotion = ref('');
      const statusText = ref('等待连接...');
      const connectionStatusText = ref('未连接');
      const conversationHistory = ref([{ sender: 'assistant', text: '您好，我是语音助手，请点击"连接服务器"开始对话。' }]);
      const logs = ref([]);
      const audioQueue = ref([]);
      const isPlaying = ref(false);
      const connectionStatusClass = ref('disconnected');
      const progressWidth = ref('0%');
      const progressText = ref('0%');
      const audioPlayer = ref(null);
      const currentAudio = ref(null);
  
      // Computed properties for button states
      const connectBtnDisabled = computed(() => isConnected.value);
      const disconnectBtnDisabled = computed(() => !isConnected.value);
      const createHistoryBtnDisabled = computed(() => !isConnected.value);
      const startRecordBtnDisabled = computed(() => !(isConnected.value && historyId.value && selectedModel.value) || isRecording.value);
      const stopRecordBtnDisabled = computed(() => !isRecording.value);
  
      // WebSocket connection function
      const connectWebSocket = () => {
        try {
          const wsUrl = apiUrl.value.trim();
  
          if (!wsUrl.startsWith('ws://') && !wsUrl.startsWith('wss://')) {
            addToLog("错误：WebSocket URL必须以ws://或wss://开头");
            updateConnectionStatus("URL错误", 'disconnected');
            return;
          }
  
          updateStatus('正在连接服务器...');
          updateConnectionStatus('正在连接...', 'processing');
          addToLog(`尝试连接到: ${wsUrl}`);
  
          // 确保先关闭之前的连接
          if (socket.value) {
            socket.value.close();
            socket.value = null;
          }
  
          socket.value = new WebSocket(wsUrl);
          
          // 添加更详细的错误处理
          socket.value.addEventListener('error', (event) => {
              console.error('WebSocket Error:', event);
              addToLog(`WebSocket详细错误信息: ${JSON.stringify(event)}`);
          });
  
          socket.value.onopen = () => {
              console.log('WebSocket连接已打开');
              addToLog('WebSocket连接已打开');
              updateStatus('已连接到服务器');
              updateConnectionStatus('已连接', 'connected');
              isConnected.value = true;
              fetchAvailableModels();
              startHeartbeat();
          };
  
          socket.value.onclose = (event) => {
            const reason = event.reason ? event.reason : "未提供关闭原因";
            const code = event.code;
            addToLog(`WebSocket连接已关闭 - 代码: ${code}, 原因: ${reason}`);
            
            // 如果连接失败且未达到最大重试次数，则重试
            if (!isConnected.value && retryCount < maxRetries) {
              retryCount++;
              addToLog(`尝试重新连接 (${retryCount}/${maxRetries})...`);
              setTimeout(tryConnect, 1000); // 1秒后重试
              return;
            }
  
            updateConnectionStatus('未连接', 'disconnected');
            updateStatus('连接已关闭');
            resetConnectionState();
          };
  
          socket.value.onerror = (error) => {
            console.error('WebSocket Error:', error);
            addToLog(`WebSocket错误: ${error}`);
            // 如果是连接阶段的错误，让 onclose 处理重试
            if (!isConnected.value) {
              return;
            }
            updateStatus('连接错误');
            updateConnectionStatus('连接错误', 'disconnected');
            resetConnectionState();
          };
  
          socket.value.onmessage = handleServerMessage;
  
        } catch (error) {
          console.error('连接错误:', error);
          addToLog(`详细错误信息: ${error.toString()}`);
          updateStatus(`连接失败: ${error.message}`);
          updateConnectionStatus('连接失败', 'disconnected');
        }
      };
  
      // WebSocket disconnection function
      const disconnectWebSocket = () => {
        stopHeartbeat();
  
        if (socket.value && socket.value.readyState === WebSocket.OPEN) {
          addToLog('断开连接');
          socket.value.close();
          updateStatus('已断开连接');
          updateConnectionStatus('已断开', 'disconnected');
        }
        resetConnectionState();
      };
  
      // Update connection status
      const updateConnectionStatus = (message, status) => {
        connectionStatusText.value = message;
        connectionStatusClass.value = status;
      };
  
      // Reset connection state
      const resetConnectionState = () => {
        isConnected.value = false;
        if (isRecording.value) stopRecording();
  
        progressWidth.value = '0%';
        progressText.value = '0%';
      };
  
      // Set session parameters
      const setSessionParams = () => {
        if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
          addToLog('WebSocket未连接');
          return;
        }
  
        if (!historyId.value) {
          addToLog('警告: 未设置对话历史ID');
          return;
        }
  
        const model = selectedModel.value;
        if (!model) {
          addToLog('警告: 未选择模型');
          return;
        }
  
        const params = {
          command: 'set_params',
          model: model,
          history_id: historyId.value,
          user_id: 'demo_user',
          provider: provider.value,
          api_base: apiBase.value
        };
  
        if (character.value.trim()) {
          params.character = character.value.trim();
        }
  
        if (emotion.value.trim()) {
          params.emotion = emotion.value.trim();
        }
  
        socket.value.send(JSON.stringify(params));
  
      };
  
      // Fetch available models
      const fetchAvailableModels = async () => {
        try {
          const baseUrl = window.location.protocol + '//' + window.location.hostname + ':8000';
          addToLog(`正在从 ${baseUrl}/llm/available-models 获取模型列表...`);
  
          const response = await fetch(`${baseUrl}/llm/available-models`);
  
          if (response.ok) {
            const data = await response.json();
            const models = data['available-models'];
  
            availableModels.value = models;
  
            selectedModel.value = models.length > 0 ? models[0] : '';
  
            if (models.length > 0) {
              addToLog(`已加载 ${models.length} 个可用模型`);
            } else {
              addToLog('未找到可用模型');
            }
          } else {
            addToLog(`获取模型失败: ${response.status} ${response.statusText}`);
          }
        } catch (error) {
          addToLog(`获取模型出错: ${error}`);
        }
      };
  
      // Create new history
      const createNewHistory = async () => {
        try {
          const baseUrl = window.location.protocol + '//' + window.location.hostname + ':8000';
          addToLog(`正在请求 ${baseUrl}/llm/history/new 创建新对话...`);
  
          const response = await fetch(`${baseUrl}/llm/history/new`, {
            method: 'POST'
          });
  
          if (response.ok) {
            const data = await response.json();
            historyId.value = data.history_id;
  
            addToLog(`创建了新对话，ID: ${historyId.value}`);
  
            setSessionParams();
          } else {
            addToLog(`创建对话失败: ${response.status} ${response.statusText}`);
          }
        } catch (error) {
          addToLog(`创建对话出错: ${error}`);
        }
      };
  
      // Start voice chat
      const startVoiceChat = () => {
        if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
          addToLog('WebSocket未连接');
          return;
        }
  
        if (!historyId.value) {
          addToLog('请先创建一个对话或输入对话历史ID');
          statusText.value = '请先创建一个对话或输入对话历史ID';
          return;
        }
  
        if (!selectedModel.value) {
          addToLog('请选择一个模型');
          statusText.value = '请选择一个模型';
          return;
        }
  
        const durationValue = parseInt(duration.value);
        const model = selectedModel.value;
  
        setSessionParams();
  
        progressWidth.value = '0%';
        progressText.value = '0%';
  
        currentStreamMessage.value = null;
  
        updateStatus('准备录音...');
  
        addMessageToConversation('正在录音...', 'user');
  
        socket.value.send(JSON.stringify({
          command: 'start_voice_chat',
          duration: durationValue,
          model: model,
          history_id: historyId.value,
          server_url: "ws://127.0.0.1:8765"
        }));
  
        isRecording.value = true;
      };
  
      // Stop recording
      const stopRecording = () => {
        if (!socket.value || socket.value.readyState !== WebSocket.OPEN || !isRecording.value) {
          return;
        }
  
        addToLog('手动停止录音');
        socket.value.send(JSON.stringify({ command: 'stop_recording' }));
  
        updateStatus('识别中...');
      };
  
      // Handle server message
      const handleServerMessage = async (event) => {
        try {
          lastServerActivity.value = Date.now();
  
          const message = JSON.parse(event.data);
  
          if (message.pong === true || message.type === 'ping' ||
            message.type === 'keep_alive' || message.type === 'audio_ready') {
            console.log('收到心跳响应');
            return;
          }
  
          const logContent = message.content ?
            `${JSON.stringify(message).substring(0, 100)}...` :
            JSON.stringify(message).substring(0, 150);
          addToLog(`收到: ${logContent}`);
  
          if (message.type === 'connection' && message.client_id) {
            clientId.value = message.client_id;
            addToLog(`已分配客户端ID: ${clientId.value}`);
          }
  
          switch (message.type) {
            case 'connection':
              updateStatus(`已连接: ${message.message}`);
              break;
  
            case 'params':
            case 'config_success':
              if (message.status === "updated" || message.message) {
                addToLog('配置已更新');
              }
              break;
  
            case 'recording':
              handleRecordingMessage(message);
              break;
  
            case 'partial_result':
              updateStatus(`识别中: ${message.text}`);
              break;
  
            case 'recognition_result':
            case 'asr_result':
              const text = message.text || '';
              updateStatus(`已识别: ${text.substring(0, 30)}${text.length > 30 ? '...' : ''}`);
              replaceUserMessage(text);
              break;
  
            case 'llm_request_start':
            case 'llm_stream_start':
              updateStatus('LLM处理中...');
              isProcessingLlm.value = true;
  
              if (!document.querySelector('.assistant-response-building')) {
                addMessageToConversation('', 'assistant', 'assistant-response-building');
              }
  
              currentStreamBuffer.value = '';
              break;
  
            case 'llm_response_chunk':
            case 'llm_stream_chunk':
              const content = message.chunk || message.content || '';
              if (content) {
                appendToLastMessage(content);
                currentStreamBuffer.value += content;
              }
              break;
  
            case 'llm_response_complete':
            case 'llm_stream_end':
              updateStatus('LLM处理完成，准备生成语音...');
              isProcessingLlm.value = false;
  
              const buildingResponse = document.querySelector('.assistant-response-building');
              if (buildingResponse) {
                buildingResponse.classList.remove('assistant-response-building');
              }
  
              if (message.history_id && (!historyId.value || historyId.value !== message.history_id)) {
                historyId.value = message.history_id;
                addToLog(`更新对话历史ID: ${historyId.value}`);
              }
  
              if (currentStreamBuffer.value) {
                const fullResponse = currentStreamBuffer.value;
                addToLog(`LLM响应完成，字数: ${fullResponse.length}`);
  
                socket.value.send(JSON.stringify({
                  command: 'llm_stream_end',
                  text: fullResponse
                }));
              } else if (message.text) {
                addToLog(`LLM响应完成，字数: ${message.text.length}`);
                socket.value.send(JSON.stringify({
                  command: 'llm_stream_end',
                  text: message.text
                }));
              } else {
                addToLog('警告：未能收集到完整的LLM响应文本');
              }
              break;
  
            case 'llm_response':
              updateStatus('LLM响应已完成');
              const building = document.querySelector('.assistant-response-building');
              if (building) {
                building.textContent = message.text;
                building.classList.remove('assistant-response-building');
              } else {
                addMessageToConversation(message.text, 'assistant');
              }
  
              if (message.audio_url) {
                playAudioResponse(message.audio_url);
              }
  
              if (message.history_id && (!historyId.value || historyId.value !== message.history_id)) {
                historyId.value = message.history_id;
                addToLog(`更新对话历史ID: ${historyId.value}`);
              }
              break;
  
            case 'tts_start':
              updateStatus('正在生成语音...');
              addToLog('TTS处理开始');
              break;
  
            case 'tts_complete':
              updateStatus('语音生成完成');
              if (message.audio_url) {
                addToLog(`TTS处理完成，音频URL: ${message.audio_url}`);
                playAudioResponse(message.audio_url);
  
                setTimeout(() => {
                  if (socket.value && socket.value.readyState === WebSocket.OPEN) {
                    sendHeartbeat();
                  }
                }, 500);
              } else {
                addToLog('TTS处理完成，但未返回音频URL');
              }
  
              return;
  
            case 'tts_sentence_complete':
              // 将新的音频添加到队列
              audioQueue.value.push({
                url: window.location.protocol + '//' + window.location.hostname + ':8000' + message.audio_url,
                text: message.text
              });
              
              // 如果没有正在播放，开始播放
              if (!isPlaying.value) {
                playNextAudio();
              }
              break;
  
            case 'error':
              updateStatus(`错误: ${message.message}`);
              addToLog(`错误: ${message.message}`);
              console.error('Server Error:', message.message);
              isProcessingLlm.value = false;
              isRecording.value = false;
              break;
  
            default:
              console.log('Received message:', message);
          }
  
        } catch (error) {
          console.error('Error handling message:', error, event.data);
          addToLog(`处理消息出错: ${error}`);
        }
      };
  
      // Handle recording message
      const handleRecordingMessage = (message) => {
        switch (message.status) {
          case 'started':
            isRecording.value = true;
            updateStatus('录音已开始...');
            break;
  
          case 'progress':
            const percentage = message.percentage;
            progressWidth.value = `${percentage}%`;
            progressText.value = `${percentage}%`;
            updateStatus(`录音中... ${message.current}秒/${message.total}秒`);
            break;
  
          case 'info':
            updateStatus(message.message);
            addToLog(message.message);
            break;
  
          case 'stopped':
            isRecording.value = false;
            updateStatus('录音已手动停止');
            break;
  
          case 'completed':
            isRecording.value = false;
            updateStatus('录音完成，等待语音识别...');
            break;
        }
      };
  
      // Update status
      const updateStatus = (message) => {
        statusText.value = message;
        console.log('Status:', message);
      };
  
      // Add message to conversation
      const addMessageToConversation = (text, sender, className = '') => {
        conversationHistory.value.push({ sender: sender, text: text, className: className });
      };
  
      // Replace last user message
      const replaceUserMessage = (text) => {
        const userMessages = conversationHistory.value.filter(message => message.sender === 'user');
        if (userMessages.length > 0) {
          userMessages[userMessages.length - 1].text = text;
        } else {
          addMessageToConversation(text, 'user');
        }
      };
  
      // Append to last message
      const appendToLastMessage = (text) => {
        const lastMessage = conversationHistory.value.find(message => message.className === 'assistant-response-building');
        if (lastMessage) {
          lastMessage.text += text;
        }
      };
  
      // Play audio response
      const playAudioResponse = (url) => {
        const baseUrl = window.location.protocol + '//' + window.location.hostname + ':8000';
        const fullUrl = baseUrl + url;
        
        // 检查队列中是否已存在相同URL的音频
        if (audioQueue.value.some(audio => audio.url === fullUrl)) {
            addToLog(`跳过重复音频: ${url}`);
            return;
        }
        
        // 添加调试日志
        console.log('准备播放音频:', fullUrl);
        addToLog(`添加音频到播放队列: ${url}`);
        
        // 添加到播放队列
        audioQueue.value.push({
            url: fullUrl,
            text: '语音回复'
        });
        
        // 如果没有正在播放，开始播放
        if (!isPlaying.value) {
            playNextAudio();
        }
      };
  
      // Add log
      const addToLog = (message) => {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        logs.value.push(`[${timeStr}] ${message}`);
      };
  
      // Start heartbeat
      const startHeartbeat = () => {
        stopHeartbeat();
  
        heartbeatInterval.value = setInterval(() => {
          if (socket.value && socket.value.readyState === WebSocket.OPEN) {
            if (Date.now() - lastServerActivity.value > 30000) {
              sendHeartbeat();
            }
          } else {
            stopHeartbeat();
          }
        }, 15000);
  
        addToLog('心跳检测已启动');
      };
  
      // Send heartbeat
      const sendHeartbeat = () => {
        try {
          socket.value.send(JSON.stringify({
            keep_alive: true,
            timestamp: Date.now()
          }));
          console.log('发送心跳...');
        } catch (e) {
          addToLog(`发送心跳失败: ${e.message}`);
          console.error('心跳发送失败', e);
        }
      };
  
      // Stop heartbeat
      const stopHeartbeat = () => {
        if (heartbeatInterval.value) {
          clearInterval(heartbeatInterval.value);
          heartbeatInterval.value = null;
        }
      };
  
      // Handle audio ended event
      const onAudioEnded = () => {
        addToLog('音频播放完成');
        
        // 移除已播放的音频
        audioQueue.value.shift();
        isPlaying.value = false;
        currentAudio.value = null;
        
        // 发送播放完成消息
        if (socket.value && socket.value.readyState === WebSocket.OPEN) {
          socket.value.send(JSON.stringify({
            command: "playback_complete",
            timestamp: Date.now()
          }));
        }
        
        // 继续播放队列中的下一个
        playNextAudio();
      };
  
      // Handle audio error event
      const onAudioError = (error) => {
        console.error('音频播放错误:', error);
        addToLog(`音频播放错误: ${error.message || '未知错误'}`);
        
        if (socket.value && socket.value.readyState === WebSocket.OPEN) {
          socket.value.send(JSON.stringify({
            command: "playback_error",
            error: error.message || '未知错误',
            timestamp: Date.now()
          }));
        }
        
        // 跳过错误的音频
        isPlaying.value = false;
        audioQueue.value.shift();
        currentAudio.value = null;
        
        // 尝试播放下一个
        playNextAudio();
      };
  
      // Handle audio load start event
      const onAudioLoadStart = () => {
        if (socket.value && socket.value.readyState === WebSocket.OPEN) {
          socket.value.send(JSON.stringify({
            keep_alive: true,
            playback_starting: true,
            timestamp: Date.now()
          }));
        }
      };
  
      // Handle audio can play event
      const onAudioCanPlay = () => {
        if (socket.value && socket.value.readyState === WebSocket.OPEN) {
          socket.value.send(JSON.stringify({
            keep_alive: true,
            playback_loaded: true,
            timestamp: Date.now()
          }));
        }
      };
  
      // Play next audio in queue
      const playNextAudio = () => {
        if (audioQueue.value.length > 0 && !isPlaying.value) {
            // 检查并移除重复音频
            if (audioQueue.value.length > 1) {
                const uniqueUrls = new Set();
                audioQueue.value = audioQueue.value.filter(audio => {
                    if (uniqueUrls.has(audio.url)) {
                        return false;
                    }
                    uniqueUrls.add(audio.url);
                    return true;
                });
            }
            
            currentAudio.value = audioQueue.value[0];
            isPlaying.value = true;
            
            try {
                if (audioPlayer.value) {
                    audioPlayer.value.src = currentAudio.value.url;
                    // 添加延迟确保音频加载完成
                    setTimeout(() => {
                        audioPlayer.value.play().catch(error => {
                            console.error('播放音频失败:', error);
                            onAudioError(error);
                        });
                    }, 100);
                }
            } catch (error) {
                console.error('设置音频源失败:', error);
                onAudioError(error);
            }
        }
      };
  
      // Handle history id change
      const onHistoryIdChange = () => {
        if (historyId.value.trim()) {
          historyId.value = historyId.value.trim();
          addToLog(`手动设置对话历史ID: ${historyId.value}`);
          setSessionParams();
        }
      };
  
      // Lifecycle hook
      onMounted(() => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          updateStatus('您的浏览器不支持录音功能');
          addToLog('错误: 浏览器不支持录音功能');
          return;
        }
  
        addToLog('页面已加载，请点击"连接"按钮连接到WebSocket服务器');
      });
  
      return {
        socket,
        clientId,
        isRecording,
        isConnected,
        isProcessingLlm,
        historyId,
        availableModels,
        recordingInterval,
        currentStreamMessage,
        currentStreamBuffer,
        heartbeatInterval,
        lastServerActivity,
        apiUrl,
        duration,
        selectedModel,
        provider,
        apiBase,
        character,
        emotion,
        statusText,
        connectionStatusText,
        conversationHistory,
        logs,
        audioQueue,
        isPlaying,
        connectionStatusClass,
        progressWidth,
        progressText,
        connectBtnDisabled,
        disconnectBtnDisabled,
        createHistoryBtnDisabled,
        startRecordBtnDisabled,
        stopRecordBtnDisabled,
        connectWebSocket,
        disconnectWebSocket,
        setSessionParams,
        fetchAvailableModels,
        createNewHistory,
        startVoiceChat,
        stopRecording,
        handleServerMessage,
        handleRecordingMessage,
        updateStatus,
        addMessageToConversation,
        replaceUserMessage,
        appendToLastMessage,
        playAudioResponse,
        addToLog,
        startHeartbeat,
        sendHeartbeat,
        stopHeartbeat,
        onAudioEnded,
        onAudioError,
        onAudioLoadStart,
        onAudioCanPlay,
        playNextAudio,
        onHistoryIdChange,
        audioPlayer,
        currentAudio
      };
    }
  };
  </script>
  
  <style scoped>
  
  body {
    font-family: 'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
  }
  
  .container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  }
  
  h1,
  h2 {
    color: #333;
    text-align: center;
  }
  
  .card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .controls {
    display: flex;
    flex-direction: column;
    gap: 15px;
    margin-bottom: 20px;
  }
  
  .btn-group {
    display: flex;
    gap: 10px;
    justify-content: center;
  }
  
  button {
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
  }
  
  .primary {
    background-color: #4CAF50;
    color: white;
  }
  
  .secondary {
    background-color: #f44336;
    color: white;
  }
  
  .neutral {
    background-color: #2196F3;
    color: white;
  }
  
  button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
  
  .conversation {
    margin-top: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    height: 300px;
    overflow-y: auto;
    padding: 10px;
    background-color: #f9f9f9;
  }
  
  .status {
    text-align: center;
    margin: 10px 0;
    font-weight: bold;
    color: #555;
  }
  
  .connected {
    color: #4CAF50;
  }
  
  .disconnected {
    color: #f44336;
  }
  
  .processing {
    color: #ff9800;
  }
  
  .user-message,
  .assistant-message {
    background-color: #E3F2FD;
    margin: 5px 0;
    margin-left: auto;
    margin-right: 0;
    text-align: right;
  }
  
  .audio-queue {
    position: relative;
    min-height: 50px;
  }
  
  .playing-status {
    padding: 10px;
    margin: 5px 0;
    background-color: #E3F2FD;
    border-radius: 4px;
    text-align: center;
  }
  
  </style>
