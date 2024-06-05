
var grid;
var drawflow;
var pluginTypes;
var stateListeners = {};
var ignore_connection_events=false;



function addGrid() {
  grid = GridStack.init({'column': 12, 'cellHeight': 50});

  /*grid.on('resize', function(event, el) {
    fitGridWidgetContents(el);
  });*/

  addGraphEditor();

  /*const html = `
    <div class="bg-success" style="font-size: 100%; overflow-y: scroll; flex-grow: 1;">I'm not a coder but if I understand it correctly when the width of the browser goes below 768x then the CSS gets switched to the one-column-mode and in doing so the added styling is lost. If I wanted to fix this so whenever the CSS changes it will refresh or "call again the liveStreamElement to add the styling back" could you tell me which file of gridstack I would need to edit?</div>
    <div class="bg-light" style="max-height: 50px">SUBMIT BUTTON</div>
  `;
  
  addGridWidget('test', 'TEST', html);*/

  drawflow = new Drawflow(document.getElementById("drawflow"));
  drawflow.start();
  
  drawflow.on('connectionCreated', onConnectionCreated);
}

function addGridWidget(id, title, html, titlebar_html, grid_options) {
  if( grid_options == undefined ) 
      grid_options = {w: 3, h: 3};

  if( titlebar_html != undefined )
      titlebar_html = `<span class="float-end float-top">${titlebar_html}</span>`;
   
  let title_html = '';
  
  if( title != undefined || titlebar_html != undefined ) {
    title_html += `<span class="mb-2">`;
    
    if( title != undefined )
      title_html += `<h5 class="d-inline">${title}</h5>`;
      
    if( titlebar_html != undefined )
      title_html += titlebar_html
      
    title_html += `</span>`;
  }    
  
  /*let card_html = `
  <div class="card ms-1 mt-1">
    <div class="card-body d-flex flex-column h-100">
      <div class="card-title" data-bs-toggle="collapse" href="${id}" role="button">
        <h5 class="d-inline">${title}</h5>
        <span class="float-end float-top fa fa-chevron-circle-down fa-lg mt-1 ms-1 me-1" data-bs-toggle="collapse" href="#${id}" id="${id}_collapse_btn" title="Minimize"></span>
        ${titlebar_html}
      </div>
      <div class="collapse show d-flex flex-column h-100" id="${id}">
          ${html}
      </div>
    </div>
  </div>
  `;*/

  let card_html = `
  <div class="card" id="${id}">
    <div class="card-body d-inline-flex flex-column h-100">
      ${title_html}
      ${html}
    </div>
  </div>
  `;
  
  let widget = grid.addWidget(card_html, grid_options);
  //fitGridWidgetContents(widget);
  return widget;
} 

function addTextInputWidget(name, id) {
  const input_id = `${id}_input`;
  const submit_id = `${id}_submit`;
  
  const html = `
    <div class="input-group">
      <textarea id="${input_id}" class="form-control" rows="2" placeholder="Enter to send (Shift+Enter for newline)"></textarea>
      <span id="${submit_id}" class="input-group-text bg-light-gray bi bi-arrow-return-left" style="color: #eeeeee;"></span>
    </div>
  `;
  
  let widget = addGridWidget(id, null, html, null, {x: 1, y: 1, w: 5, h: 2});

  let onsubmit = function() {
    const input = document.getElementById(input_id);
    console.log('submitting text input', input.value);
    msg = {}
    msg[name] = {'input': input.value};
    sendWebsocket(msg);
    input.value = "";
  }
  
  let onkeydown = function(event) {
    // https://stackoverflow.com/a/49389811
    if( event.which === 13 && !event.shiftKey ) {
      if( !event.repeat )
        onsubmit();
      event.preventDefault(); // prevents the addition of a new line in the text field
    }
  }

  document.getElementById(input_id).addEventListener('keydown', onkeydown);
  document.getElementById(submit_id).addEventListener('click', onsubmit);

  return widget;
}

function addChatWidget(name, id) {
  const history_id = `${id}_history`;
  const input_id = `${id}_input`;
  const submit_id = `${id}_submit`;
  
  const html = `
    <div id="${history_id}" class="bg-medium-gray p-2 mb-2" style="font-size: 100%; overflow-y: scroll; flex-grow: 1;" ondrop="onFileDrop(event)" ondragover="onFileDrag(event)"></div>
    <div class="input-group">
      <textarea id="${input_id}" class="form-control" rows="2" placeholder="Enter to send (Shift+Enter for newline)"></textarea>
      <span id="${submit_id}" class="input-group-text bg-light-gray bi bi-arrow-return-left" style="color: #eeeeee;"></span>
    </div>
  `;
  
  let widget = addGridWidget(id, name, html, null, {x: 1, y: 1, w: 5, h: 6});

  let onsubmit = function() {
    const input = document.getElementById(input_id);
    console.log('submitting chat message:', input.value);
    sendWebsocket({'UserPrompt': {'input': input.value}});
    input.value = "";
  }
  
  let onkeydown = function(event) {
    // https://stackoverflow.com/a/49389811
    if( event.which === 13 && !event.shiftKey ) {
      if( !event.repeat )
        onsubmit();
      event.preventDefault(); // prevents the addition of a new line in the text field
    }
  }

  document.getElementById(input_id).addEventListener('keydown', onkeydown);
  document.getElementById(submit_id).addEventListener('click', onsubmit);

  addStateListener(name, function(state_dict) {
    console.log(`${name}.stateListener()`, state_dict);
    if( 'history' in state_dict ) {
      updateChatHistory(history_id, state_dict['history']);
    }
  });
  
  sendWebsocket({'get_state_dict': name});
  return widget;
}

function updateChatHistory(id, history) {    
    let chj = $(`#${id}`);
    let chc = document.getElementById(id);
    let isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;

    chj.empty(); // clear because server may remove partial/rejected ASR prompts
    
    for( let n=0; n < history.length; n++ ) {
      const role = history[n]['role'];
      
      /*if( role == 'system' )
        continue;*/
        
      let contents = '';
      var hasImage = 'image' in history[n];
      
      if( hasImage ) {
        contents += `<img src=${history[n]['image']} width="100%">`;
        imageAtBottom = true;
      }
      
      if( 'text' in history[n] )
        contents += history[n]['text'];

      if( contents.length > 0 )
        chj.append(`<div id="msg_${n}" class="chat-message-${role} mb-3">${contents}</div><br/>`);
    }

    function scrollBottom(container) {  // https://stackoverflow.com/a/21067431
      container.scrollTop = container.scrollHeight - container.clientHeight;
      console.log(`scrolling to bottom ${container.scrollTop} ${container.scrollHeight} ${container.clientHeight}`);
    }

    if( isScrolledToBottom ) { // autoscroll unless the user has scrolled up
      if( hasImage )
        setTimeout(scrollBottom, 50, chc);  // wait for images to load to get right height
      else
        scrollBottom(chc);
    }
}


function addVideoOutputWidget(name, id) {
  const video_id = `${id}_video_player`;
  const html = `
    <div>
      <video id="${video_id}" autoplay controls playsinline muted>Your browser does not support video</video>
    </div>
  `;
  
  let widget = addGridWidget(id, name, html, null, {x: 0, y: 0, w: 8, h: 5});
  let video = document.getElementById(video_id);
  
  video.addEventListener('playing', function(e) {
    const abs_rect = video.getBoundingClientRect();
    const options = {'w': Math.ceil(abs_rect.width/grid.cellWidth()), 'h': Math.ceil(abs_rect.height/grid.getCellHeight())+1};
    console.log(`${video_id} playing`, e, abs_rect, options, grid.cellWidth(), grid.getCellHeight());
    grid.update(widget, options);
  });

  playStream(getWebsocketURL('output'), video); // TODO handle actual stream name
  
  return widget;
}


function addStateListener(name, listener) {
  if( name == undefined )
    name = 'all';
    
  if( ! (name in stateListeners) )
    stateListeners[name] = [listener];
  else
    stateListeners[name].push(listener);
}

function setStateDict(state_dicts) {
  for( plugin_name in state_dicts ) {
    const state_dict = state_dicts[plugin_name];
    for( state_name in state_dict ) {
      const id = document.getElementById(`${plugin_name}_config_${state_name}`);
      if( id != null ) {
        id.value = state_dict[state_name];
      }
    }
    
    if( plugin_name in stateListeners )
      stateListeners[plugin_name].forEach((listener) => {listener(state_dict);});

    if( 'all' in stateListeners )
      stateListeners['all'].forEach((listener) => {listener(state_dict);});
  }
}

function setStats(stats_dicts) {
  for( plugin_name in stats_dicts ) {
    const stats = stats_dicts[plugin_name];
    
    if( ! 'summary' in stats )
      continue;
    
    let summary = stats['summary'];
    
    if( Array.isArray(summary) )
      summary = summary.join('<br/>');
        
    document.getElementById(`${plugin_name}_node_stats`).innerHTML = summary;
  }
}


function addPlugin(plugin) {
  console.log('addPlugin() =>');
  console.log(plugin)
  
  const plugin_name = plugin['name'];
  const nodes = document.querySelectorAll('.drawflow-node');
  
  let x = 10;
  let y = 10;
    
  if( nodes.length > 0 ) {
    const node = nodes[nodes.length-1];
    const abs_rect = node.getBoundingClientRect();
    x = node.offsetLeft + abs_rect.width + 60;
    y = node.offsetTop
  }

  const html = `
    <div style="position: absolute; top: 5px;">
      ${plugin_name}
      <p id="${plugin_name}_node_stats" style="font-size: 80%"></p>
    </div>
    <!--<div style="font-size: 80%">ABC123</div>-->
  `;
  
  drawflow.addNode(plugin_name, plugin['inputs'].length, plugin['outputs'].length, x, y, plugin_name, {}, html);
  
  let style = '';
  
  for( i in plugin['outputs'] ) {
    const output = plugin['outputs'][i].toString();
    style += `.${plugin_name} .outputs .output:nth-child(${Number(i)+1}):before {`;
    style += `display: block; content: "${output}"; position: relative; min-width: 160px; font-size: 80%; bottom: 2px; right: ${(output.length-1) * 6 + 15}px;} `;
  }
    
  style += `.outputs { margin-top: 20px; } `;
  
  var style_el = document.createElement('style');
  style_el.id = `${plugin_name}_node_io_styles`;
  style_el.innerHTML = style;
  document.head.appendChild(style_el);

  $(`.${plugin_name}`).on('dblclick', function () {
    console.log(`double-click ${plugin_name}`);
    if( addPluginGridWidget(plugin_name, plugin['type']) == null ) {
      sendWebsocket({'get_state_dict': plugin_name});
      const config_modal = new bootstrap.Modal(`#${plugin['name']}_config_dialog`);
      config_modal.show();
    }
  });
  
  addPluginDialog(plugin_name, 'config', null, plugin['parameters']);
}

function addPluginGridWidget(name, type) {
  const id = `${name}_grid`;
  
  if( document.getElementById(id) != null )
    return null;
    
  switch(type) {
    case 'UserPrompt':
      return addTextInputWidget(name, id);
    case 'ChatSession':
      return addChatWidget(name, id);
    case 'VideoOutput':
      return addVideoOutputWidget(name, id);
    default:
      return null;
  }
}

function addPlugins(plugins) {
  if( !Array.isArray(plugins) )
    plugins = [plugins];
    
  for( i in plugins )
    addPlugin(plugins[i]);

  ignore_connection_events = true;
  
  for( i in plugins ) {
    for( l in plugins[i]['connections'] ) {
      const conn = plugins[i]['connections'][l];
      console.log('adding connection', conn);

      drawflow.addConnection(
        drawflow.getNodesFromName(plugins[i]['name'])[0], 
        drawflow.getNodesFromName(conn['to'])[0],  
        `output_${conn['output']+1}`, 
        `input_${conn['input']+1}`
      );
    }
  }
  
  ignore_connection_events = false;
}

function addPluginTypes(types) {
  pluginTypes = types;
  
  let id = document.getElementById("add_plugin_menu_list");
            
  if( id != null )
    id.innerHTML = pluginMenuList();

  for( pluginName in pluginTypes ) {
    const plugin = pluginTypes[pluginName];
    let pluginMenu = document.getElementById(`menu_create_plugin_${pluginName}`);
    
    if( 'init' in plugin && Object.keys(plugin['init']['parameters']).length > 0 ) {
      const dialog_id = addPluginDialog(pluginName, 'init', plugin['init']['description'], plugin['init']['parameters']);
      pluginMenu.addEventListener('click', (event) => {
        console.log(`opening dialog ${dialog_id}`);
        const modal = new bootstrap.Modal(`#${dialog_id}`);
        modal.show();
      });
    }
    else {
      pluginMenu.addEventListener('click', (event) => {
        sendWebsocket({
          'init_plugin': {
            'name': plugin['name'],
            'args': {}
          }
        });
      });
    }
  }
}

function pluginMenuList() {
  if( pluginTypes == undefined )
    return '';
 
  let html = '';
  
  for( plugin in pluginTypes )
    html += `<li><a class="dropdown-item" id="menu_create_plugin_${plugin}" href="#">${plugin}</a></li>`; // data-bs-toggle="modal" data-bs-target="#${plugin}_init_dialog"

  return html; 
}  

function addGraphEditor() {

  let titlebar_html = `
    <span class="dropdown float-end float-top">
      <button class="btn btn-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">
        Add
      </button>
      <ul class="dropdown-menu" id="add_plugin_menu_list">
        ${pluginMenuList()}
      </ul>
    </span>
  `;
  
  let html = `
    <div id="drawflow" class="mt-1 flex-grow-1 w-100"></div>
  `;

  let widget = addGridWidget(
    "graph_editor", "Graph Editor", 
     html, titlebar_html, 
     {x: 0, y: 0, w: 8, h: 11} /*, noMove: true}*/
  );
  
  let drawflow = document.getElementById("drawflow");
  
  drawflow.addEventListener('mouseenter', (event) => {
    console.log('onmouseleave(drawflow) => disabling grid move/resize');
    grid.disable();
  });
  
  drawflow.addEventListener('mouseleave', (event) => {
    console.log('onmouseleave(drawflow) => enabling grid move/resize');
    grid.enable();
  });
  
  return widget;
}

function addDialog(id, title, html, onsubmit, oncancel) {
  $('body').append(`
  <div class="modal fade" id="${id}" tabindex="-1" aria-labelledby="${id}_label" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="${id}_label">${title}</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close" style="color: #eeeeee;"></button>
          </div>
          <div class="modal-body">
            ${html}
          </div>
          <div class="modal-footer">
            <button id="${id}_cancel" type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button> 
            <button id="${id}_submit" type="button" class="btn btn-secondary" data-bs-dismiss="modal">Okay</button>
          </div>
        </div>
      </div>
    </div>
  `);
  
  if( onsubmit != undefined )
    $(`#${id}_submit`).on('click', onsubmit);
    
  if( oncancel != undefined )
    $(`#${id}_cancel`).on('click', oncancel);
}

function onConnectionCreated(event) {
  console.log(`onConnectionCreated(ignore=${ignore_connection_events})`, event);
  
  if( ignore_connection_events )
    return;
    
  sendWebsocket({
    'add_connection': {
      'input': {
        'name': drawflow.getNodeFromId(event['input_id'])['name'],
        'channel': Number(event['input_class'].split('_').at(-1)) - 1
      },
      'output': {
        'name': drawflow.getNodeFromId(event['output_id'])['name'],
        'channel': Number(event['output_class'].split('_').at(-1)) - 1
      }
    }
  });
}

function addPluginDialog(plugin_name, stage, description, parameters) {

  let html = '';
  
  if( description != null )
    html += `<p>${description}</p>`;
    
  html += '<form>';

  for( param_name in parameters ) {
    const param = parameters[param_name];
    const id = `${plugin_name}_${stage}_${param_name}`;
    
    let value = '';
    
    if( 'default' in param && param['default'] != undefined )
      value = `value="${param['default']}"`;
      
    let help = '';
    
    if( 'help' in param )
      help = `<div id="${id}_help" class="form-text">${param['help']}</div>`;
    
    switch(param['type']) {
      case 'number':
      case 'integer':
        var type='number'; break;
      default:
        var type='text';
    }
    
    html += `
      <div class="mb-3">
        <label for="${id}" class="form-label">${param['display_name']}</label>
        <input id="${id}" type="${type}" class="form-control" aria-describedby="${id}_help" ${value}>
        ${help}
      </div>
    `;
  }
  
  html += `</form>`;
  
  let onsubmit = function() {
    console.log(`onsubmit(${plugin_name})`);
    let args = {};
    
    for( param_name in parameters ) {
      let value = document.getElementById(`${plugin_name}_${stage}_${param_name}`).value;
      if( value != undefined && value.length > 0 ) { // input.value are always strings
        const type = parameters[param_name]['type'];
        if( type == 'integer' || type == 'number' )
          value = Number(value);
        args[param_name] = value;
      }
    }
    
    console.log(`${stage}Plugin(${plugin_name}) =>`);
    console.log(args);

    let msg = {};
    
    msg[`${stage}_plugin`] = {
        'name': plugin_name,
        'args': args
    }
    
    sendWebsocket(msg);
  }
  
  const dialog_id = `${plugin_name}_${stage}_dialog`;
  addDialog(dialog_id, plugin_name, html, onsubmit);
  return dialog_id;
}

function fitGridWidgetContents(el) {
  //let width = parseInt(el.getAttribute('gs-w')) || 0;
  //let height = parseInt(el.getAttribute('gs-h')) || 0;
  //console.log('grid widget resize', width, height, el.offsetWidth, el.offsetHeight);
  //let card_title = el.querySelector('.card-title');
  //console.log('grid widget resize', card_title, card_title.offsetHeight);
  //el.querySelector('.collapse').style.height = `${el.offsetHeight - card_title.offsetHeight - 50}px`;
  console.log('grid widget resize', el.offsetWidth, el.offsetHeight);
  
  let grid_rect = el.getBoundingClientRect();

  let card = el.querySelector('.card-body');
  let card_rect = card.getBoundingClientRect();
  
  let card_inner = el.querySelector('.collapse');
  let card_inner_rect = card_inner.getBoundingClientRect();
  
  console.log(`grid widget rect height=${grid_rect.height}  bottom=${grid_rect.bottom}`);
  console.log(`card widget rect height=${card_rect.height}  bottom=${card_rect.bottom}`);
  console.log(`card inner rect height=${card_inner_rect.height}  bottom=${card_inner_rect.bottom}`);
  
  //if( card_inner_rect.bottom > card_rect.bottom ) {
    //card_inner.style.height = `${card_inner_rect.height - (card_inner_rect.bottom - card_rect.bottom)}px`;
    //console.log('set height', card_inner.style.height);
  //}
  /*if( card_rect.height > grid_rect.height ) {
    //card.style.height = `${grid_rect.height}px`;
    card_inner.style.height = `${card_inner.offsetHeight - (card_rect.height - grid_rect.height)}px`;
  }*/
  //card.style.height = `${el.offsetHeight}px`;
  //el.querySelector('.collapse').style.height = `${el.offsetHeight-150}px`;
}


